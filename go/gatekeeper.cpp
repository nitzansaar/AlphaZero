/*
 * gatekeeper.cpp — Single-process gate worker.
 *
 * Plays a contiguous slice of gate games [game_offset, game_offset+num_games).
 * Color alternates by absolute game index so results aggregate correctly across
 * multiple worker processes.
 *
 * Invoked by gatekeeper_runner.py (one process per CPU core), which collects
 * the "WINS: X" line from stdout and makes the final accept/reject decision.
 *
 * Usage (normally via gatekeeper_runner.py):
 *   ./gatekeeper <new_ts.pt> <best_ts.pt> <new_iter> <best_iter> [options]
 *
 * Options:
 *   --games N          games this worker plays      (default: 100)
 *   --sims  N          MCTS sims per move            (default: 400)
 *   --batch N          MCTS leaf batch size          (default: 1)
 *   --game-offset N    starting absolute game index  (default: 0)
 *   --temp-moves N     high-temp moves per game      (default: 4)
 *   --max-moves N      force-end after N moves       (default: 300)
 *   --min-pass-move N  suppress pass before move N   (default: 30)
 *   --cuda             use GPU
 *   --seed N           base RNG seed                 (default: 42)
 *
 * Stdout: "WINS: X\n" — wins for the new model out of games played.
 * Stderr: per-game progress lines.
 *
 * Compile:
 *   make gatekeeper
 */

#include "nn_inference.h"
#include "mcts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

/* ── Process-local NN pointers ───────────────────────────────────────────*/

static NNInference *proc_new_nn  = nullptr;
static NNInference *proc_best_nn = nullptr;

static void new_nn_cb(const float *planes, int batch_size,
                      float *values, float *policies)
{
    proc_new_nn->eval(planes, batch_size, values, policies);
}

static void best_nn_cb(const float *planes, int batch_size,
                       float *values, float *policies)
{
    proc_best_nn->eval(planes, batch_size, values, policies);
}

/* ── Config ──────────────────────────────────────────────────────────────*/

struct Config {
    std::string new_model_path;
    std::string best_model_path;
    int         new_iter      = 0;
    int         best_iter     = 0;
    int         num_games     = 100;
    int         num_sims      = 400;
    int         batch_size    = 1;
    int         game_offset   = 0;    /* absolute game index of first game */
    int         temp_moves    = 4;
    int         max_moves     = 300;
    int         min_pass_move = 30;
    bool        use_cuda      = false;
    uint32_t    seed          = 42;
    std::string worker_dir    = "";   /* if set, write progress/wins files here */
};

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s <new_ts.pt> <best_ts.pt> <new_iter> <best_iter> [options]\n"
            "  --games N, --sims N, --batch N, --game-offset N\n"
            "  --temp-moves N, --max-moves N, --min-pass-move N\n"
            "  --cuda, --seed N\n",
            prog);
}

static bool parse_args(int argc, char *argv[], Config &cfg)
{
    if (argc < 5) return false;
    cfg.new_model_path  = argv[1];
    cfg.best_model_path = argv[2];
    cfg.new_iter        = atoi(argv[3]);
    cfg.best_iter       = atoi(argv[4]);

    for (int i = 5; i < argc; i++) {
        if      (!strcmp(argv[i], "--games")         && i+1<argc) cfg.num_games     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sims")          && i+1<argc) cfg.num_sims      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch")         && i+1<argc) cfg.batch_size    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--game-offset")   && i+1<argc) cfg.game_offset   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp-moves")    && i+1<argc) cfg.temp_moves    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-moves")     && i+1<argc) cfg.max_moves     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-pass-move") && i+1<argc) cfg.min_pass_move = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cuda"))                       cfg.use_cuda      = true;
        else if (!strcmp(argv[i], "--seed")          && i+1<argc) cfg.seed          = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--worker-dir")    && i+1<argc) cfg.worker_dir    = argv[++i];
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); return false; }
    }
    return true;
}

/* ── main ────────────────────────────────────────────────────────────────*/

int main(int argc, char *argv[])
{
    Config cfg;
    if (!parse_args(argc, argv, cfg)) { print_usage(argv[0]); return 1; }

    go_init();

    NNInference new_nn (cfg.new_model_path,  cfg.use_cuda);
    NNInference best_nn(cfg.best_model_path, cfg.use_cuda);
    proc_new_nn  = &new_nn;
    proc_best_nn = &best_nn;

    mcts_seed_rng(cfg.seed + (uint32_t)cfg.game_offset * 7u);

    NodePool *pool = new NodePool;
    int wins = 0;

    for (int local = 0; local < cfg.num_games; local++) {
        int game_idx      = cfg.game_offset + local;
        int new_model_abs = (game_idx % 2 == 0) ? 1 : -1;  /* alternate colors */

        fprintf(stderr, "  game %d/%d  (new=%s)\n",
                game_idx + 1, cfg.game_offset + cfg.num_games,
                new_model_abs == 1 ? "Black" : "White");

        GoState state       = go_initial_state();
        int absolute_player = 1;
        int move_count      = 0;

        while (!go_game_ended(&state) && move_count < cfg.max_moves) {
            NNEvalFn cb = (absolute_player == new_model_abs) ? new_nn_cb : best_nn_cb;

            mcts_init_root(pool, &state, absolute_player);
            mcts_simulate(pool, cb, cfg.num_sims, cfg.batch_size, /*noise=*/false);

            float probs[ACTION_SIZE];
            float temp   = (move_count < cfg.temp_moves) ? 1.0f : 0.0f;
            int   action = mcts_select_move(pool, temp, probs);

            if (action == PASS_ACTION && move_count < cfg.min_pass_move) {
                probs[PASS_ACTION] = 0.0f;
                float total = 0.0f;
                for (int i = 0; i < ACTION_SIZE; i++) total += probs[i];
                if (total > 0.0f)
                    for (int i = 0; i < ACTION_SIZE; i++) probs[i] /= total;
                action = 0;
                for (int i = 1; i < PASS_ACTION; i++)
                    if (probs[i] > probs[action]) action = i;
            }

            state           = go_next_state_canonical(&state, action);
            absolute_player = -absolute_player;
            move_count++;
        }

        int  winner  = go_get_winner(&state, absolute_player);
        bool new_won = (winner == new_model_abs);
        if (new_won) wins++;

        const int BAR    = 20;
        int  filled      = ((local + 1) * BAR) / cfg.num_games;
        char bar[BAR+1];
        for (int i = 0; i < BAR; i++) bar[i] = i < filled ? '#' : '.';
        bar[BAR] = '\0';

        fprintf(stderr, "  [%s] %s  wins=%d/%d\n",
                bar, new_won ? "WIN " : "LOSS", wins, local + 1);

        /* Write progress files polled by gatekeeper_runner.py */
        if (!cfg.worker_dir.empty()) {
            auto write_file = [&](const char *name, int val) {
                std::string p = cfg.worker_dir + "/" + name;
                if (FILE *fp = fopen(p.c_str(), "w")) {
                    fprintf(fp, "%d\n", val);
                    fclose(fp);
                }
            };
            write_file("progress", local + 1);
            write_file("wins",     wins);
        }
    }

    delete pool;

    /* Machine-readable result for gatekeeper_runner.py */
    fprintf(stdout, "WINS: %d\n", wins);
    fflush(stdout);
    return 0;
}
