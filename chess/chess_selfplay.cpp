/*
 * chess_selfplay.cpp — AlphaZero-style self-play data generator for chess.
 *
 * Produces three .npy files in an output directory:
 *   states.npy    float32 (N, 19, 8, 8)        AlphaZero 19-plane representation
 *   policies.npy  float32 (N, 4672)            MCTS visit-count probabilities
 *   values.npy    float32 (N,)                 game outcome per position
 *
 * Usage:
 *   chess_selfplay <model_ts.pt> [options]
 *   chess_selfplay output_chess/models/5/model_ts.pt --games 50 --sims 400 --threads 4
 *
 * Adapted from go/selfplay_cpp.cpp.  Differences: 19-plane single-timestep
 * input (no move history), chess rules engine, chess result interpretation.
 */

#include "nn_inference.h"
#include "mcts.h"
#include "chess_encoding.h"
#include "npy_writer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include <random>

using Clock = std::chrono::steady_clock;
using Dsec  = std::chrono::duration<double>;

struct Timings {
    double mcts_simulation = 0.0;
    double move_selection  = 0.0;
    double state_copy      = 0.0;
    long long total_game_moves = 0;
    int    completed_games = 0;
    int    checkmates      = 0;
    int    draws           = 0;
    int    force_ended     = 0;

    void operator+=(const Timings &o) {
        mcts_simulation += o.mcts_simulation;
        move_selection  += o.move_selection;
        state_copy      += o.state_copy;
        total_game_moves += o.total_game_moves;
        completed_games += o.completed_games;
        checkmates      += o.checkmates;
        draws           += o.draws;
        force_ended     += o.force_ended;
    }
};

/* ── Thread-local NN pointer ─────────────────────────────────────────── */

static thread_local NNInference *tl_nn = nullptr;

static void nn_callback(const float *planes, int batch_size,
                        float *values, float *policies)
{
    tl_nn->eval(planes, batch_size, values, policies);
}

/* ── Configuration ────────────────────────────────────────────────────── */

struct Config {
    std::string model_path;
    int      num_games   = 100;
    int      num_sims    = 400;
    int      batch_size  = 32;
    int      num_threads = 1;
    bool     use_cuda    = false;
    std::string output_dir = ".";
    int      temp_moves  = 30;    /* high-temp (explore) moves per game     */
    float    final_temp  = 0.1f;  /* temperature after temp_moves           */
    int      max_moves   = 200;   /* force-end (draw) if exceeded           */
    uint32_t seed        = 42;
    float    full_prob   = 1.0f;  /* fraction of turns using full search    */
    int      fast_sims   = 100;   /* sims for non-training (fast) turns      */
    float    c_puct          = 1.414f;
    float    dirichlet_alpha = CHESS_DIR_ALPHA;
    float    dirichlet_frac  = CHESS_DIR_FRAC;
};

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s <model_ts.pt> [options]\n\n"
        "Options:\n"
        "  --games N        total games to play         (default: 100)\n"
        "  --sims  N        MCTS sims per move          (default: 400)\n"
        "  --batch N        MCTS leaf batch size        (default: 32)\n"
        "  --threads N      parallel worker threads     (default: 1)\n"
        "  --cuda           use GPU if available\n"
        "  --output DIR     output directory            (default: .)\n"
        "  --temp-moves N   high-temp moves per game    (default: 30)\n"
        "  --final-temp F   temperature after temp-moves(default: 0.1)\n"
        "  --max-moves N    force-end (draw) cap        (default: 200)\n"
        "  --seed N         base RNG seed               (default: 42)\n"
        "  --full-prob F    fraction of turns w/ full search (default: 1.0)\n"
        "  --fast-sims N    sims for non-training turns       (default: 100)\n"
        "  --c-puct F            MCTS exploration constant    (default: 1.414)\n"
        "  --dirichlet-alpha F   root noise alpha             (default: 0.3)\n"
        "  --dirichlet-frac F    root noise mix fraction      (default: 0.25)\n",
        prog);
}

static bool parse_args(int argc, char *argv[], Config &cfg)
{
    if (argc < 2) return false;
    cfg.model_path = argv[1];
    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--games")    && i+1 < argc) cfg.num_games   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sims")     && i+1 < argc) cfg.num_sims    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch")    && i+1 < argc) cfg.batch_size  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")  && i+1 < argc) cfg.num_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cuda"))                   cfg.use_cuda    = true;
        else if (!strcmp(argv[i], "--output")   && i+1 < argc) cfg.output_dir  = argv[++i];
        else if (!strcmp(argv[i], "--temp-moves") && i+1 < argc) cfg.temp_moves = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--final-temp") && i+1 < argc) cfg.final_temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-moves")  && i+1 < argc) cfg.max_moves  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")       && i+1 < argc) cfg.seed       = (uint32_t)strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--full-prob")  && i+1 < argc) cfg.full_prob  = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--fast-sims")  && i+1 < argc) cfg.fast_sims  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--c-puct")     && i+1 < argc) cfg.c_puct     = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--dirichlet-alpha") && i+1 < argc) cfg.dirichlet_alpha = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--dirichlet-frac")  && i+1 < argc) cfg.dirichlet_frac  = (float)atof(argv[++i]);
        else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); return false; }
    }
    if (cfg.c_puct <= 0.0f)        { fprintf(stderr, "--c-puct must be > 0\n"); return false; }
    if (cfg.dirichlet_alpha < 0.0f){ fprintf(stderr, "--dirichlet-alpha must be >= 0\n"); return false; }
    if (cfg.dirichlet_frac < 0.0f || cfg.dirichlet_frac > 1.0f) {
        fprintf(stderr, "--dirichlet-frac must be in [0,1]\n"); return false; }
    if (cfg.batch_size > MAX_BATCH_SIZE) {
        fprintf(stderr, "--batch must be <= %d\n", MAX_BATCH_SIZE); return false; }
    return true;
}

/* ── One training example ─────────────────────────────────────────────── */

struct Step {
    float planes[CHESS_NUM_PLANES * 64];
    float probs[CHESS_ACTION_SIZE];
    int   absolute_player;            /* +1 White / -1 Black */
};

/* ── Result interpretation ────────────────────────────────────────────────
 *
 * Returns +1 White win, -1 Black win, 0 draw for a terminal board.  At a
 * checkmate the side to move is mated and loses.
 */
static int board_winner(const chess::Board &board)
{
    auto over = board.isGameOver();
    if (over.first == chess::GameResultReason::CHECKMATE) {
        int stm = (board.sideToMove() == chess::Color::WHITE) ? 1 : -1;
        return -stm;   /* side to move is checkmated → opponent wins */
    }
    return 0;          /* stalemate / draws */
}

/* ── Single-game self-play ────────────────────────────────────────────── */

static void play_one_game(NodePool           *pool,
                          const Config       &cfg,
                          std::vector<float> &out_states,
                          std::vector<float> &out_policies,
                          std::vector<float> &out_values,
                          std::mt19937       &coin_rng,
                          Timings            &t)
{
    std::vector<Step> steps;
    steps.reserve((size_t)cfg.max_moves);

    chess::Board board;                       /* start position, White to move */
    int move_count = 0;

    std::bernoulli_distribution full_dist(cfg.full_prob);

    while (board.isGameOver().first == chess::GameResultReason::NONE &&
           move_count < cfg.max_moves) {

        int absolute_player = (board.sideToMove() == chess::Color::WHITE) ? 1 : -1;

        bool is_full = full_dist(coin_rng);
        int  sims    = is_full ? cfg.num_sims : cfg.fast_sims;

        Step step;
        step.absolute_player = absolute_player;

        auto tp = Clock::now();
        chess_board_to_planes(board, step.planes);
        t.state_copy += Dsec(Clock::now() - tp).count();

        auto tm = Clock::now();
        mcts_set_c_puct(cfg.c_puct);
        mcts_init_root(pool, board);
        mcts_simulate(pool, nn_callback, sims, cfg.batch_size,
                      /*add_noise=*/is_full, cfg.dirichlet_alpha, cfg.dirichlet_frac);
        t.mcts_simulation += Dsec(Clock::now() - tm).count();

        auto ts = Clock::now();
        float temp = (move_count < cfg.temp_moves) ? 1.0f : cfg.final_temp;
        chess::Move chosen;
        int action = mcts_select_move(pool, temp, step.probs, &chosen);
        t.move_selection += Dsec(Clock::now() - ts).count();

        if (action < 0) break;                /* no legal moves (shouldn't happen) */

        if (is_full)
            steps.push_back(step);

        board.makeMove(chosen);
        move_count++;
    }

    bool terminal = (board.isGameOver().first != chess::GameResultReason::NONE);
    int winner = terminal ? board_winner(board) : 0;   /* force-end → draw */

    t.completed_games++;
    t.total_game_moves += move_count;
    if (!terminal)            t.force_ended++;
    else if (winner == 0)     t.draws++;
    else                      t.checkmates++;

    for (const Step &s : steps) {
        float value = (winner == 0) ? 0.0f
                    : (winner == s.absolute_player) ? 1.0f : -1.0f;
        out_states.insert(out_states.end(), s.planes, s.planes + CHESS_NUM_PLANES * 64);
        out_policies.insert(out_policies.end(), s.probs, s.probs + CHESS_ACTION_SIZE);
        out_values.push_back(value);
    }
}

/* ── Worker thread ────────────────────────────────────────────────────── */

static void worker(int thread_id, int num_games, const Config &cfg,
                   std::vector<float> &shared_states,
                   std::vector<float> &shared_policies,
                   std::vector<float> &shared_values,
                   std::mutex &out_mutex, Timings &shared_timings)
{
    NNInference nn(cfg.model_path, cfg.use_cuda);
    tl_nn = &nn;

    mcts_seed_rng(cfg.seed + (uint32_t)thread_id * 1000u);
    std::mt19937 coin_rng(cfg.seed + (uint32_t)thread_id * 1000u + 7919u);

    NodePool *pool = new NodePool;

    std::vector<float> local_states, local_policies, local_values;
    Timings local_t;

    for (int g = 0; g < num_games; g++) {
        play_one_game(pool, cfg, local_states, local_policies, local_values,
                      coin_rng, local_t);

        {   /* progress file polled by the Python runner */
            std::string pp = cfg.output_dir + "/progress";
            if (FILE *fp = fopen(pp.c_str(), "w")) {
                fprintf(fp, "%d\n", g + 1);
                fclose(fp);
            }
        }
        if ((g + 1) % 10 == 0 || g + 1 == num_games)
            fprintf(stderr, "[thread %d] %d/%d games done\n", thread_id, g + 1, num_games);
    }

    delete pool;

    std::lock_guard<std::mutex> lock(out_mutex);
    shared_states.insert(shared_states.end(), local_states.begin(), local_states.end());
    shared_policies.insert(shared_policies.end(), local_policies.begin(), local_policies.end());
    shared_values.insert(shared_values.end(), local_values.begin(), local_values.end());
    shared_timings += local_t;
}

/* ── main ─────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    Config cfg;
    if (!parse_args(argc, argv, cfg)) { print_usage(argv[0]); return 1; }

    fprintf(stderr, "=== chess_selfplay ===\n");
    fprintf(stderr, "Model     : %s\n", cfg.model_path.c_str());
    fprintf(stderr, "Games     : %d\n", cfg.num_games);
    fprintf(stderr, "Sims      : %d\n", cfg.num_sims);
    fprintf(stderr, "Batch     : %d\n", cfg.batch_size);
    fprintf(stderr, "Threads   : %d\n", cfg.num_threads);
    fprintf(stderr, "CUDA      : %s\n", cfg.use_cuda ? "yes" : "no");
    fprintf(stderr, "Output    : %s\n", cfg.output_dir.c_str());
    fprintf(stderr, "TempMoves : %d (final temp %.2f)\n", cfg.temp_moves, cfg.final_temp);
    fprintf(stderr, "MaxMoves  : %d\n", cfg.max_moves);
    fprintf(stderr, "FullProb  : %.3f  FastSims: %d\n", cfg.full_prob, cfg.fast_sims);
    fprintf(stderr, "C_PUCT    : %.3f\n", cfg.c_puct);
    fprintf(stderr, "DirNoise  : alpha=%.4f frac=%.3f\n\n",
            cfg.dirichlet_alpha, cfg.dirichlet_frac);

    std::vector<float> all_states, all_policies, all_values;
    std::mutex out_mutex;
    Timings all_timings;

    if (cfg.num_threads <= 1) {
        worker(0, cfg.num_games, cfg, all_states, all_policies, all_values,
               out_mutex, all_timings);
    } else {
        int base = cfg.num_games / cfg.num_threads;
        int rem  = cfg.num_games % cfg.num_threads;
        std::vector<std::thread> threads;
        for (int tt = 0; tt < cfg.num_threads; tt++) {
            int n = base + (tt < rem ? 1 : 0);
            threads.emplace_back(worker, tt, n, std::cref(cfg),
                                 std::ref(all_states), std::ref(all_policies),
                                 std::ref(all_values), std::ref(out_mutex),
                                 std::ref(all_timings));
        }
        for (auto &th : threads) th.join();
    }

    int N = (int)all_values.size();
    fprintf(stderr, "\nTotal positions: %d\n", N);

    {
        std::string path = cfg.output_dir + "/states.npy";
        int dims[] = {N, CHESS_NUM_PLANES, 8, 8};
        if (npy_write_float32(path.c_str(), all_states.data(),
                              N * CHESS_NUM_PLANES * 64, 4, dims) != 0) {
            fprintf(stderr, "ERROR writing %s\n", path.c_str()); return 1; }
        fprintf(stderr, "Wrote: %s\n", path.c_str());
    }
    {
        std::string path = cfg.output_dir + "/policies.npy";
        int dims[] = {N, CHESS_ACTION_SIZE};
        if (npy_write_float32(path.c_str(), all_policies.data(),
                              N * CHESS_ACTION_SIZE, 2, dims) != 0) {
            fprintf(stderr, "ERROR writing %s\n", path.c_str()); return 1; }
        fprintf(stderr, "Wrote: %s\n", path.c_str());
    }
    {
        std::string path = cfg.output_dir + "/values.npy";
        int dims[] = {N};
        if (npy_write_float32(path.c_str(), all_values.data(), N, 1, dims) != 0) {
            fprintf(stderr, "ERROR writing %s\n", path.c_str()); return 1; }
        fprintf(stderr, "Wrote: %s\n", path.c_str());
    }
    {
        std::string path = cfg.output_dir + "/timing.json";
        if (FILE *fp = fopen(path.c_str(), "w")) {
            fprintf(fp, "{\n  \"timings\": {\n");
            fprintf(fp, "    \"mcts_simulation\": %.6f,\n", all_timings.mcts_simulation);
            fprintf(fp, "    \"move_selection\": %.6f,\n",  all_timings.move_selection);
            fprintf(fp, "    \"state_copy\": %.6f\n",       all_timings.state_copy);
            fprintf(fp, "  },\n  \"metrics\": {\n");
            fprintf(fp, "    \"completed_games\": %d,\n",   all_timings.completed_games);
            fprintf(fp, "    \"total_game_moves\": %lld,\n", all_timings.total_game_moves);
            fprintf(fp, "    \"checkmates\": %d,\n",        all_timings.checkmates);
            fprintf(fp, "    \"draws\": %d,\n",             all_timings.draws);
            fprintf(fp, "    \"force_ended\": %d\n",        all_timings.force_ended);
            fprintf(fp, "  }\n}\n");
            fclose(fp);
            fprintf(stderr, "Wrote: %s\n", path.c_str());
        }
    }

    return 0;
}
