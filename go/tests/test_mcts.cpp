/*
 * tests/test_mcts.cpp
 *
 * Compile:
 *   g++ -O2 -std=c++17 -Wall -I.. -o test_mcts \
 *       ../go_engine.c ../mcts.cpp test_mcts.cpp && ./test_mcts
 */

#include "mcts.h"

#include <cstdio>
#include <cmath>
#include <cstring>

/* ── Test harness ─────────────────────────────────────────────────────── */

static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define EXPECT(cond, msg) do {                               \
    tests_run++;                                             \
    if (cond) {                                              \
        tests_passed++;                                      \
    } else {                                                 \
        tests_failed++;                                      \
        printf("  FAIL [line %d]: %s\n", __LINE__, (msg));  \
    }                                                        \
} while (0)

static void print_summary()
{
    printf("\n=== Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0)
        printf("  (%d FAILED)", tests_failed);
    printf(" ===\n");
}

/* ── Stub neural networks ─────────────────────────────────────────────── */

/* Uniform policy, value = 0 (drawn game prediction). */
static void nn_uniform(const float *, int batch_size,
                       float *values, float *policies)
{
    for (int b = 0; b < batch_size; b++) {
        values[b] = 0.0f;
        for (int a = 0; a < ACTION_SIZE; a++)
            policies[b * ACTION_SIZE + a] = 1.0f / (float)ACTION_SIZE;
    }
}

/* Always predicts current player wins (value = +1). */
static void nn_always_win(const float *, int batch_size,
                          float *values, float *policies)
{
    for (int b = 0; b < batch_size; b++) {
        values[b] = 1.0f;
        for (int a = 0; a < ACTION_SIZE; a++)
            policies[b * ACTION_SIZE + a] = 1.0f / (float)ACTION_SIZE;
    }
}

/* ── Shared pool — static to avoid stack overflow (~44 MB) ───────────── */

static NodePool s_pool;

/* ── Tests ────────────────────────────────────────────────────────────── */

static void test_init_root()
{
    printf("\n[init_root]\n");
    GoState s   = go_initial_state();
    int root    = mcts_init_root(&s_pool, &s, 1);

    EXPECT(root == 0,                         "root is always index 0");
    EXPECT(s_pool.next_free == 1,             "one node allocated after init");
    EXPECT(s_pool.nodes[0].player == 1,       "root player = 1 (Black)");
    EXPECT(s_pool.nodes[0].visits == 0,       "root starts with 0 visits");
    EXPECT(s_pool.nodes[0].state_set == 1,    "root state is set");
    EXPECT(s_pool.nodes[0].num_children == 0, "root starts with no children");
    EXPECT(s_pool.nodes[0].parent_idx == -1,  "root has no parent");
}

static void test_root_expanded_after_simulate()
{
    printf("\n[root expanded after first simulate]\n");
    GoState s = go_initial_state();
    mcts_init_root(&s_pool, &s, 1);

    /* A single simulation triggers root expansion. */
    mcts_simulate(&s_pool, nn_uniform, 1, 1, false);

    /* Empty 9×9 board: all 82 actions valid → 82 children. */
    EXPECT(s_pool.nodes[0].num_children == ACTION_SIZE,
           "root should have ACTION_SIZE children on empty board");
}

static void test_root_visits_equal_num_sims()
{
    printf("\n[root visits == num_simulations]\n");
    GoState s = go_initial_state();
    mcts_init_root(&s_pool, &s, 1);

    int N = 50;
    mcts_simulate(&s_pool, nn_uniform, N, 8, false);

    EXPECT(s_pool.nodes[0].visits == N,
           "root visits should equal num_simulations");
}

static void test_child_visit_sum_equals_root_visits()
{
    printf("\n[sum of child visits == root visits]\n");
    GoState s = go_initial_state();
    mcts_init_root(&s_pool, &s, 1);

    int N = 200;
    mcts_simulate(&s_pool, nn_uniform, N, 16, false);

    /* Every simulation path passes through exactly one direct child of
     * the root, so the total direct-child visit count must equal N. */
    const Node *root = &s_pool.nodes[0];
    int child_sum = 0;
    for (int a = 0; a < ACTION_SIZE; a++) {
        int ci = root->children[a];
        if (ci >= 0) child_sum += s_pool.nodes[ci].visits;
    }
    EXPECT(child_sum == N,
           "sum of root child visits should equal num_simulations");
}

static void test_all_root_children_visited()
{
    printf("\n[all root children visited after ACTION_SIZE sims]\n");
    GoState s = go_initial_state();
    mcts_init_root(&s_pool, &s, 1);

    /* With uniform prior, Q=0 (stub value=0) and equal priors, UCB always
     * favours the least-visited child.  After 82 sims with batch_size=1
     * every root child should have been visited at least once. */
    mcts_simulate(&s_pool, nn_uniform, ACTION_SIZE, 1, false);

    const Node *root = &s_pool.nodes[0];
    int unvisited = 0;
    for (int a = 0; a < ACTION_SIZE; a++) {
        int ci = root->children[a];
        if (ci >= 0 && s_pool.nodes[ci].visits == 0) unvisited++;
    }
    EXPECT(unvisited == 0,
           "after ACTION_SIZE sims every root child should be visited");
}

static void test_no_pool_overflow_800_sims()
{
    printf("\n[no pool overflow – 800 sims]\n");
    GoState s = go_initial_state();
    mcts_init_root(&s_pool, &s, 1);

    mcts_simulate(&s_pool, nn_uniform, 800, 32, false);

    EXPECT(s_pool.next_free < NODE_POOL_SIZE,
           "800 sims should not overflow the node pool");
    printf("  nodes used: %d / %d  (%.1f%%)\n",
           s_pool.next_free, NODE_POOL_SIZE,
           100.0f * s_pool.next_free / NODE_POOL_SIZE);
}

static void test_all_visited_nodes_have_states()
{
    printf("\n[all visited nodes have states computed]\n");
    GoState s = go_initial_state();
    mcts_init_root(&s_pool, &s, 1);
    mcts_simulate(&s_pool, nn_uniform, 50, 8, false);

    int missing = 0;
    for (int i = 0; i < s_pool.next_free; i++) {
        const Node *n = &s_pool.nodes[i];
        if (n->visits > 0 && !n->state_set) missing++;
    }
    EXPECT(missing == 0,
           "every visited node should have its state computed");
}

static void test_terminal_root_not_expanded()
{
    printf("\n[terminal root not expanded]\n");
    GoState s            = go_initial_state();
    s.consecutive_passes = 2;   /* game over — two passes */

    mcts_init_root(&s_pool, &s, 1);
    mcts_simulate(&s_pool, nn_uniform, 10, 4, false);

    EXPECT(s_pool.nodes[0].num_children == 0,
           "terminal root should not be expanded");
}

static void test_terminal_backup_uses_actual_winner()
{
    printf("\n[terminal backup uses actual winner]\n");

    /* Set up a clearly won position for Black. */
    GoState s = go_initial_state();
    for (int i = 0; i < 50; i++) s.board[i] =  1;   /* black stones */
    for (int i = 50; i < 60; i++) s.board[i] = -1;  /* white stones */
    s.consecutive_passes = 2;   /* game over */

    /* Confirm Black wins. */
    int winner = go_get_winner(&s, 1);
    EXPECT(winner == 1, "Black should win this position");

    /* Root is Black's turn (player=1), Black wins → Q should be +1. */
    mcts_init_root(&s_pool, &s, 1);
    mcts_simulate(&s_pool, nn_uniform, 10, 4, false);

    const Node *root = &s_pool.nodes[0];
    if (root->visits > 0) {
        float q = root->total_value / (float)root->visits;
        EXPECT(q > 0.0f,
               "root Q should be positive when Black wins terminal position");
    }
}

static void test_select_move_exploit()
{
    printf("\n[select_move – exploit (temperature=0)]\n");
    GoState s = go_initial_state();
    mcts_init_root(&s_pool, &s, 1);
    mcts_simulate(&s_pool, nn_uniform, 200, 16, false);

    /* Find the most-visited root child by direct inspection. */
    const Node *root = &s_pool.nodes[0];
    int best_action = -1, best_visits = -1;
    for (int a = 0; a < ACTION_SIZE; a++) {
        int ci = root->children[a];
        if (ci < 0) continue;
        if (s_pool.nodes[ci].visits > best_visits) {
            best_visits = s_pool.nodes[ci].visits;
            best_action = a;
        }
    }

    float probs[ACTION_SIZE];
    int selected = mcts_select_move(&s_pool, 0.0f, probs);

    EXPECT(selected == best_action,
           "temperature=0 should select most-visited child");
}

static void test_action_probs_sum_to_one()
{
    printf("\n[action_probs sum to 1]\n");
    GoState s = go_initial_state();
    mcts_init_root(&s_pool, &s, 1);
    mcts_simulate(&s_pool, nn_uniform, 100, 16, false);

    float probs[ACTION_SIZE];
    mcts_select_move(&s_pool, 1.0f, probs);

    float sum = 0.0f;
    for (int a = 0; a < ACTION_SIZE; a++) sum += probs[a];

    EXPECT(fabsf(sum - 1.0f) < 1e-4f, "action_probs should sum to 1.0");
}

static void test_nn_value_sign_propagates_correctly()
{
    printf("\n[NN value sign propagates correctly via backup]\n");

    /* nn_always_win returns value=+1 from the leaf player's perspective.
     *
     * Root = Black (player=+1).  First leaf = White's node (player=-1).
     * In backup, for root (player=+1):
     *   leaf_player(-1) != root.player(+1)  →  node_value = -1.
     * So root.total_value < 0 after many sims. */
    GoState s = go_initial_state();
    mcts_init_root(&s_pool, &s, 1);
    mcts_simulate(&s_pool, nn_always_win, 50, 8, false);

    const Node *root = &s_pool.nodes[0];
    EXPECT(root->visits == 50, "50 sims → root visits = 50");

    float q = root->total_value / (float)root->visits;
    EXPECT(q < 0.0f,
           "root Q should be negative when NN always says leaf player wins");
}

static void test_dirichlet_noise_changes_priors()
{
    printf("\n[Dirichlet noise changes root child priors]\n");
    GoState s = go_initial_state();

    /* Expand root WITHOUT noise; record priors. */
    mcts_init_root(&s_pool, &s, 1);
    mcts_simulate(&s_pool, nn_uniform, 0, 1, false);

    float prior_clean[ACTION_SIZE] = {};
    for (int a = 0; a < ACTION_SIZE; a++) {
        int ci = s_pool.nodes[0].children[a];
        if (ci >= 0) prior_clean[a] = s_pool.nodes[ci].prior;
    }

    /* Expand root WITH noise; record priors. */
    mcts_init_root(&s_pool, &s, 1);
    mcts_simulate(&s_pool, nn_uniform, 0, 1, true);

    float prior_noisy[ACTION_SIZE] = {};
    for (int a = 0; a < ACTION_SIZE; a++) {
        int ci = s_pool.nodes[0].children[a];
        if (ci >= 0) prior_noisy[a] = s_pool.nodes[ci].prior;
    }

    int changed = 0;
    for (int a = 0; a < ACTION_SIZE; a++)
        if (fabsf(prior_noisy[a] - prior_clean[a]) > 1e-6f) changed++;

    EXPECT(changed > 0,
           "Dirichlet noise should change at least some root priors");
}

static void test_valid_actions_only()
{
    printf("\n[only valid actions appear as root children]\n");
    GoState s = go_initial_state();

    /* Fill most of the board so only a few moves are legal. */
    for (int i = 0; i < NUM_POSITIONS - 3; i++)
        s.board[i] = (i % 2 == 0) ? 1 : -1;

    mcts_init_root(&s_pool, &s, 1);
    mcts_simulate(&s_pool, nn_uniform, 1, 1, false);

    /* All children of root must correspond to valid moves. */
    int invalid_children = 0;
    const Node *root = &s_pool.nodes[0];
    for (int a = 0; a < ACTION_SIZE; a++) {
        if (root->children[a] < 0) continue;
        if (!go_is_valid_move(&root->state, a, 1)) invalid_children++;
    }
    EXPECT(invalid_children == 0,
           "all root children should be valid moves");
}

static void test_player_alternates_down_tree()
{
    printf("\n[player alternates down the tree]\n");
    GoState s = go_initial_state();
    mcts_init_root(&s_pool, &s, 1);   /* root = Black */
    mcts_simulate(&s_pool, nn_uniform, 10, 1, false);

    /* Every root child should have player=-1 (White). */
    const Node *root = &s_pool.nodes[0];
    int wrong_player = 0;
    for (int a = 0; a < ACTION_SIZE; a++) {
        int ci = root->children[a];
        if (ci < 0) continue;
        if (s_pool.nodes[ci].player != -1) wrong_player++;
    }
    EXPECT(wrong_player == 0,
           "root children (depth 1) should all have player=-1 (White)");

    /* Every grandchild should have player=+1 (Black). */
    for (int a = 0; a < ACTION_SIZE; a++) {
        int ci = root->children[a];
        if (ci < 0) continue;
        const Node *child = &s_pool.nodes[ci];
        for (int b = 0; b < ACTION_SIZE; b++) {
            int gci = child->children[b];
            if (gci < 0) continue;
            if (s_pool.nodes[gci].player != 1) wrong_player++;
        }
    }
    EXPECT(wrong_player == 0,
           "grandchildren (depth 2) should all have player=+1 (Black)");
}

/* ── main ─────────────────────────────────────────────────────────────── */

int main()
{
    go_init();
    printf("=== mcts tests ===\n");

    test_init_root();
    test_root_expanded_after_simulate();
    test_root_visits_equal_num_sims();
    test_child_visit_sum_equals_root_visits();
    test_all_root_children_visited();
    test_no_pool_overflow_800_sims();
    test_all_visited_nodes_have_states();
    test_terminal_root_not_expanded();
    test_terminal_backup_uses_actual_winner();
    test_select_move_exploit();
    test_action_probs_sum_to_one();
    test_nn_value_sign_propagates_correctly();
    test_dirichlet_noise_changes_priors();
    test_valid_actions_only();
    test_player_alternates_down_tree();

    print_summary();
    return (tests_failed > 0) ? 1 : 0;
}
