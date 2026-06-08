#pragma once

/*
 * mcts.h — AlphaZero-style batched MCTS tree for chess.
 *
 * Mirrors the Go C++ MCTS (go/mcts.cpp) but adapted for chess:
 *
 *   - Sparse children.  Chess ACTION_SIZE is 4672; a dense int[4672] per node
 *     would make a 100k-node pool ~1.9 GB.  Each node instead stores up to
 *     MAX_LEGAL_MOVES (218) ChildEntry records.
 *
 *   - No per-node board.  chess::Board owns a heap std::vector for repetition
 *     history, so storing one per node would be ruinous.  Each node stores only
 *     the chess::Move that led to it; a leaf's board is reconstructed on demand
 *     by replaying moves from the root board (cheap relative to an NN call).
 *
 *   - 19-plane single-timestep input (no 8-step history like Go).
 *
 *   - Player convention matches the Python chess code: +1 White / -1 Black.
 *
 * Usage:
 *   NodePool *pool = new NodePool;
 *   chess::Board board;                     // current game position
 *   mcts_init_root(pool, board);            // player taken from board.sideToMove()
 *   mcts_simulate(pool, nn_fn, 400, 32, true);
 *   float probs[CHESS_ACTION_SIZE];
 *   chess::Move mv;
 *   int action = mcts_select_move(pool, 1.0f, probs, &mv);
 */

#include "chess_encoding.h"

/* ── Compile-time limits ──────────────────────────────────────────────── */

#ifndef NODE_POOL_SIZE
#  define NODE_POOL_SIZE  100000
#endif

#define MAX_BATCH_SIZE  64
#define MAX_PATH_DEPTH  256

/* ── MCTS hyper-parameters ────────────────────────────────────────────── */

extern float g_mcts_c_puct;                 /* exploration constant            */
constexpr float CHESS_DIR_ALPHA = 0.3f;     /* Dirichlet concentration (config)*/
constexpr float CHESS_DIR_FRAC  = 0.25f;    /* noise mix-in fraction           */

void mcts_set_c_puct(float c_puct);

/* ── Sparse child entry ───────────────────────────────────────────────── */

struct ChildEntry {
    int action_idx;   /* action index in [0, CHESS_ACTION_SIZE)             */
    int pool_idx;     /* index of the child Node in the pool                */
};

/* ── Node ─────────────────────────────────────────────────────────────── */

struct Node {
    float prior;          /* P(s,a) prior probability from NN                */
    int   player;         /* absolute player to move here: +1 White / -1 Black */
    int   parent_idx;     /* pool index of parent; -1 for root              */
    int   action_idx;     /* action that led to this node; -1 for root      */
    chess::Move move;     /* move that led to this node (from parent)        */
    int   visits;         /* N(s,a) visit count                             */
    float total_value;    /* W(s,a) accumulated value                       */
    int   virtual_loss;   /* temporary penalty for parallel traversals      */
    int   num_children;
    ChildEntry children[MAX_LEGAL_MOVES];
};

/* ── Node pool (bump allocator) ───────────────────────────────────────── */

/*
 * Trivially constructible POD-ish array (chess::Move is trivial), so
 * `new NodePool` is a single allocation with no per-node constructor cost.
 * root_board holds the position at the root, carrying real game history for
 * correct threefold / fifty-move detection inside the tree.
 */
struct NodePool {
    Node        nodes[NODE_POOL_SIZE];
    int         next_free;
    chess::Board root_board;
};

/* ── Neural-network evaluation callback ────────────────────────────────────
 *
 * planes:     batch_size × CHESS_NUM_PLANES × 64 floats (row-major), produced
 *             by chess_board_to_planes
 * values:     [out] batch_size floats in [-1, 1]
 * policies:   [out] batch_size × CHESS_ACTION_SIZE softmax probabilities
 */
typedef void (*NNEvalFn)(const float *planes, int batch_size,
                          float *values, float *policies);

/* ── Public API ───────────────────────────────────────────────────────── */

/*
 * Reset the pool and initialise root node 0 from `board`.  The root player is
 * board.sideToMove() (+1 White / -1 Black).  Returns 0 (the root index).
 */
int mcts_init_root(NodePool *pool, const chess::Board &board);

/*
 * Run batched MCTS simulations from the root.
 *   nn_fn           NN callback used to evaluate leaf nodes
 *   num_simulations total simulations to run
 *   batch_size      leaves collected per NN call (≤ MAX_BATCH_SIZE)
 *   add_noise       add Dirichlet noise to root child priors
 *   noise_alpha     Dirichlet concentration
 *   noise_frac      fraction of each prior replaced by sampled noise
 */
void mcts_simulate(NodePool *pool, NNEvalFn nn_fn,
                   int num_simulations, int batch_size, bool add_noise,
                   float noise_alpha = CHESS_DIR_ALPHA,
                   float noise_frac  = CHESS_DIR_FRAC);

/*
 * Select a move from the root using temperature-weighted visit counts.
 *   temperature       1.0 → sample ∝ visits; 0 → argmax visits
 *   action_probs_out  [out] CHESS_ACTION_SIZE normalised visit counts (target)
 *   chosen_move_out   [out, optional] the chess::Move for the chosen action
 * Returns the chosen action index, or -1 if the root has no children.
 */
int mcts_select_move(const NodePool *pool, float temperature,
                     float *action_probs_out,
                     chess::Move *chosen_move_out = nullptr);

/* Re-seed the thread-local RNG (Dirichlet noise + move sampling). */
void mcts_seed_rng(uint32_t seed);
