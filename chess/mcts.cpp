#include "mcts.h"

#include <cstring>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <random>

/* ── RNG (thread-local: independent stream per worker thread) ───────────── */

static thread_local std::mt19937 s_rng(42);
float g_mcts_c_puct = 1.414f;

void mcts_seed_rng(uint32_t seed) { s_rng.seed(seed); }
void mcts_set_c_puct(float c_puct) { if (c_puct > 0.0f) g_mcts_c_puct = c_puct; }

/* ── Pool helpers ─────────────────────────────────────────────────────── */

static int pool_alloc(NodePool *pool)
{
    assert(pool->next_free < NODE_POOL_SIZE &&
           "node pool exhausted — increase NODE_POOL_SIZE");
    return pool->next_free++;
}

static void node_init(Node *n, float prior, int player,
                      int parent_idx, int action_idx, chess::Move move)
{
    n->prior        = prior;
    n->player       = player;
    n->parent_idx   = parent_idx;
    n->action_idx   = action_idx;
    n->move         = move;
    n->visits       = 0;
    n->total_value  = 0.0f;
    n->virtual_loss = 0;
    n->num_children = 0;
}

/* ── Board reconstruction ─────────────────────────────────────────────────
 *
 * Replay the moves from the root board down to `leaf_idx`.  Each non-root node
 * stores the move that produced it, so walking parent links collects the move
 * sequence; replaying it on a copy of the root board yields the leaf position
 * (with full game + tree history for repetition detection).
 */
static chess::Board reconstruct_board(const NodePool *pool, int leaf_idx)
{
    chess::Move stack[MAX_PATH_DEPTH];
    int d = 0, cur = leaf_idx;
    while (pool->nodes[cur].parent_idx >= 0 && d < MAX_PATH_DEPTH) {
        stack[d++] = pool->nodes[cur].move;
        cur = pool->nodes[cur].parent_idx;
    }
    chess::Board b = pool->root_board;
    for (int k = d - 1; k >= 0; k--)
        b.makeMove(stack[k]);
    return b;
}

/* ── Public: init root ────────────────────────────────────────────────── */

int mcts_init_root(NodePool *pool, const chess::Board &board)
{
    pool->next_free = 0;
    pool->root_board = board;
    int player = (board.sideToMove() == chess::Color::WHITE) ? 1 : -1;
    int root_idx = pool_alloc(pool);   /* always 0 */
    node_init(&pool->nodes[root_idx], 0.0f, player, -1, -1,
              chess::Move(chess::Move::NO_MOVE));
    return root_idx;
}

/* ── UCB child selection ──────────────────────────────────────────────────
 *
 *   U(s,a) = -Q_adj + C * P(s,a) * sqrt(N_parent) / (1 + N_child)
 *   Q_adj  = (W - virtual_loss) / (N + virtual_loss)
 *
 * Returns the pool index of the best child, or -1 if there are none.
 */
static int select_best_child(const NodePool *pool, int node_idx)
{
    const Node *parent = &pool->nodes[node_idx];
    int   Ns     = parent->visits + parent->virtual_loss;
    float sqrtNs = sqrtf((float)Ns);

    float best_score = -FLT_MAX;
    int   best_child = -1;

    for (int i = 0; i < parent->num_children; i++) {
        int ci = parent->children[i].pool_idx;
        const Node *child = &pool->nodes[ci];

        int   Nsa = child->visits + child->virtual_loss;
        float Q = (Nsa > 0)
                ? (child->total_value - (float)child->virtual_loss) / (float)Nsa
                : 0.0f;
        float U = -Q + g_mcts_c_puct * child->prior * sqrtNs / (1.0f + (float)Nsa);

        if (U > best_score) {
            best_score = U;
            best_child = ci;
        }
    }
    return best_child;
}

/* ── Node expansion ───────────────────────────────────────────────────────
 *
 * Create a child for every legal move, with renormalised prior.  Mirrors
 * expand_node in go/mcts.cpp but uses sparse children and stores the move.
 */
static void expand_node(NodePool *pool, int node_idx, const chess::Board &board,
                        const float *policy, int next_player)
{
    int action_arr[MAX_LEGAL_MOVES];
    chess::Move move_arr[MAX_LEGAL_MOVES];
    int n = chess_legal_moves(board, action_arr, move_arr);

    /* Renormalise priors over legal moves so they sum to 1. */
    float prior_sum = 0.0f;
    for (int i = 0; i < n; i++)
        prior_sum += policy[action_arr[i]];
    float scale = (prior_sum > 0.0f) ? 1.0f / prior_sum : 0.0f;

    Node *node = &pool->nodes[node_idx];
    for (int i = 0; i < n; i++) {
        float prob = (scale > 0.0f) ? policy[action_arr[i]] * scale
                                    : 1.0f / (float)n;   /* uniform fallback */
        int child_idx = pool_alloc(pool);
        node_init(&pool->nodes[child_idx], prob, next_player,
                  node_idx, action_arr[i], move_arr[i]);
        /* node may have moved if pool_alloc reallocated? No — flat array. */
        node = &pool->nodes[node_idx];
        node->children[node->num_children].action_idx = action_arr[i];
        node->children[node->num_children].pool_idx   = child_idx;
        node->num_children++;
    }
}

/* ── Dirichlet noise ──────────────────────────────────────────────────── */

static void add_dirichlet_noise(NodePool *pool, int root_idx,
                                float noise_alpha, float noise_frac)
{
    Node *root = &pool->nodes[root_idx];
    if (root->num_children == 0) return;
    if (noise_alpha <= 0.0f || noise_frac <= 0.0f) return;
    if (noise_frac > 1.0f) noise_frac = 1.0f;

    std::gamma_distribution<float> gamma(noise_alpha, 1.0f);

    float noise[MAX_LEGAL_MOVES];
    float noise_sum = 0.0f;
    for (int i = 0; i < root->num_children; i++) {
        float g  = gamma(s_rng);
        noise[i] = g;
        noise_sum += g;
    }
    if (noise_sum <= 0.0f) return;

    float new_sum = 0.0f;
    for (int i = 0; i < root->num_children; i++) {
        Node *child = &pool->nodes[root->children[i].pool_idx];
        float nz = noise[i] / noise_sum;
        child->prior = (1.0f - noise_frac) * child->prior + noise_frac * nz;
        new_sum += child->prior;
    }
    if (new_sum > 0.0f)
        for (int i = 0; i < root->num_children; i++)
            pool->nodes[root->children[i].pool_idx].prior /= new_sum;
}

/* ── Backup ───────────────────────────────────────────────────────────── */

static void backup(NodePool  *pool,
                   const int *path,  int path_len,
                   bool       terminal, int terminal_winner,
                   int        leaf_player, float nn_value)
{
    for (int i = path_len - 1; i >= 0; i--) {
        Node *n = &pool->nodes[path[i]];
        n->visits++;

        float v;
        if (terminal) {
            v = (terminal_winner == 0) ? 0.0f
              : (terminal_winner == n->player) ? 1.0f : -1.0f;
        } else {
            v = (leaf_player == n->player) ? nn_value : -nn_value;
        }
        n->total_value += v;
    }
}

/* ── Leaf collection ──────────────────────────────────────────────────── */

struct LeafInfo {
    int leaf_idx;
    int path[MAX_PATH_DEPTH];
    int path_len;
};

static void collect_leaves(NodePool *pool, int root_idx, int batch_size,
                            LeafInfo *out, int *count_out)
{
    int count = 0;
    for (int b = 0; b < batch_size; b++) {
        int path[MAX_PATH_DEPTH];
        int path_len = 0;
        int cur      = root_idx;

        pool->nodes[cur].virtual_loss++;
        path[path_len++] = cur;

        while (pool->nodes[cur].num_children > 0) {
            int child = select_best_child(pool, cur);
            if (child < 0) break;
            cur = child;
            pool->nodes[cur].virtual_loss++;
            path[path_len++] = cur;
            if (path_len >= MAX_PATH_DEPTH) break;
        }

        LeafInfo *li = &out[count++];
        li->leaf_idx = cur;
        memcpy(li->path, path, path_len * sizeof(int));
        li->path_len = path_len;
    }
    *count_out = count;
}

/* ── Main simulation loop ─────────────────────────────────────────────── */

void mcts_simulate(NodePool *pool, NNEvalFn nn_fn,
                   int num_simulations, int batch_size, bool add_noise,
                   float noise_alpha, float noise_frac)
{
    assert(batch_size <= MAX_BATCH_SIZE);

    Node *root = &pool->nodes[0];

    /* ── Step 1: expand root ──────────────────────────────────────────── */
    {
        auto over = pool->root_board.isGameOver();
        if (over.first == chess::GameResultReason::NONE) {
            float planes[CHESS_NUM_PLANES * 64];
            chess_board_to_planes(pool->root_board, planes);

            float root_value;
            float root_policy[CHESS_ACTION_SIZE];
            nn_fn(planes, 1, &root_value, root_policy);

            expand_node(pool, 0, pool->root_board, root_policy, -root->player);

            if (add_noise)
                add_dirichlet_noise(pool, 0, noise_alpha, noise_frac);
        }
    }

    /* ── Step 2: simulation loop ──────────────────────────────────────── */
    int sims_done = 0;
    while (sims_done < num_simulations) {
        int cur_batch = batch_size;
        if (sims_done + cur_batch > num_simulations)
            cur_batch = num_simulations - sims_done;

        /* 2a — collect leaves with virtual loss applied */
        LeafInfo raw[MAX_BATCH_SIZE];
        int      raw_count = 0;
        collect_leaves(pool, 0, cur_batch, raw, &raw_count);

        /* 2b — deduplicate by pool index */
        int unique_idx[MAX_BATCH_SIZE];
        int unique_count = 0;
        int raw_to_unique[MAX_BATCH_SIZE];
        for (int i = 0; i < raw_count; i++) {
            int li = raw[i].leaf_idx;
            int j  = 0;
            for (; j < unique_count; j++)
                if (unique_idx[j] == li) break;
            if (j == unique_count) unique_idx[unique_count++] = li;
            raw_to_unique[i] = j;
        }

        /* 2c — reconstruct boards and build NN batch */
        static thread_local float batch_planes[MAX_BATCH_SIZE * CHESS_NUM_PLANES * 64];
        float batch_values  [MAX_BATCH_SIZE];
        static thread_local float batch_policies[MAX_BATCH_SIZE * CHESS_ACTION_SIZE];

        chess::Board leaf_boards[MAX_BATCH_SIZE];
        bool         leaf_terminal[MAX_BATCH_SIZE];
        int          leaf_winner[MAX_BATCH_SIZE];
        int          eval_slot[MAX_BATCH_SIZE];   /* nn batch row, or -1 */
        int          nn_batch = 0;

        for (int j = 0; j < unique_count; j++) {
            leaf_boards[j] = reconstruct_board(pool, unique_idx[j]);
            auto over = leaf_boards[j].isGameOver();
            bool terminal = (over.first != chess::GameResultReason::NONE);
            leaf_terminal[j] = terminal;

            if (terminal) {
                int leaf_player = pool->nodes[unique_idx[j]].player;
                leaf_winner[j] = (over.first == chess::GameResultReason::CHECKMATE)
                               ? -leaf_player : 0;   /* stm is checkmated → loses */
                eval_slot[j] = -1;
            } else {
                chess_board_to_planes(leaf_boards[j],
                                      batch_planes + nn_batch * CHESS_NUM_PLANES * 64);
                eval_slot[j] = nn_batch;
                leaf_winner[j] = 0;
                nn_batch++;
            }
        }

        if (nn_batch > 0)
            nn_fn(batch_planes, nn_batch, batch_values, batch_policies);

        /* 2e — remove virtual loss from all collected paths */
        for (int i = 0; i < raw_count; i++)
            for (int k = 0; k < raw[i].path_len; k++)
                pool->nodes[raw[i].path[k]].virtual_loss--;

        /* 2f — expand non-terminal leaves and back up every path */
        for (int j = 0; j < unique_count; j++) {
            int   li   = unique_idx[j];
            Node *leaf = &pool->nodes[li];

            float value = 0.0f;
            if (!leaf_terminal[j]) {
                int   ei  = eval_slot[j];
                value     = batch_values[ei];
                float *pol = batch_policies + ei * CHESS_ACTION_SIZE;
                if (leaf->num_children == 0)
                    expand_node(pool, li, leaf_boards[j], pol, -leaf->player);
            }

            for (int i = 0; i < raw_count; i++) {
                if (raw_to_unique[i] != j) continue;
                backup(pool, raw[i].path, raw[i].path_len,
                       leaf_terminal[j], leaf_winner[j], leaf->player, value);
            }
        }

        sims_done += cur_batch;
    }
}

/* ── Move selection ───────────────────────────────────────────────────── */

int mcts_select_move(const NodePool *pool, float temperature,
                     float *action_probs_out, chess::Move *chosen_move_out)
{
    const Node *root = &pool->nodes[0];

    memset(action_probs_out, 0, CHESS_ACTION_SIZE * sizeof(float));
    if (root->num_children == 0) return -1;

    /* Normalised visit counts (training target). */
    float total = 0.0f;
    for (int i = 0; i < root->num_children; i++) {
        int   ci = root->children[i].pool_idx;
        float v  = (float)pool->nodes[ci].visits;
        action_probs_out[root->children[i].action_idx] = v;
        total += v;
    }
    if (total > 0.0f)
        for (int i = 0; i < root->num_children; i++)
            action_probs_out[root->children[i].action_idx] /= total;

    int chosen_entry = -1;

    if (temperature < 1e-6f || total <= 0.0f) {
        /* Exploit: argmax visits. */
        int best_visits = -1;
        for (int i = 0; i < root->num_children; i++) {
            int v = pool->nodes[root->children[i].pool_idx].visits;
            if (v > best_visits) { best_visits = v; chosen_entry = i; }
        }
    } else {
        /* Explore: sample ∝ visits^(1/temperature). */
        float weights[MAX_LEGAL_MOVES];
        float wsum = 0.0f;
        for (int i = 0; i < root->num_children; i++) {
            float w = powf((float)pool->nodes[root->children[i].pool_idx].visits,
                           1.0f / temperature);
            weights[i] = w;
            wsum += w;
        }
        std::uniform_real_distribution<float> uniform(0.0f, wsum);
        float r = uniform(s_rng), cum = 0.0f;
        for (int i = 0; i < root->num_children; i++) {
            cum += weights[i];
            if (r <= cum) { chosen_entry = i; break; }
        }
        if (chosen_entry < 0) chosen_entry = root->num_children - 1;
    }

    if (chosen_move_out)
        *chosen_move_out = pool->nodes[root->children[chosen_entry].pool_idx].move;
    return root->children[chosen_entry].action_idx;
}
