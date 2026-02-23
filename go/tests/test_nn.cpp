/*
 * tests/test_nn.cpp
 *
 * Tests for nn_inference.h / nn_inference.cpp.
 * Requires a TorchScript model produced by export_model.py.
 *
 * Compile (from go/ directory):
 *   TORCH=$(.venv/bin/python -c "import torch,os; print(os.path.dirname(torch.__file__))")
 *   g++ -O2 -std=c++17 -fno-pie -no-pie -I. \
 *       -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
 *       -L$TORCH/lib -Wl,-rpath,$TORCH/lib \
 *       -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10 \
 *       -o test_nn \
 *       go_engine.c mcts.cpp nn_inference.cpp tests/test_nn.cpp
 *
 * Run:
 *   BOARD_SIZE=9 .venv/bin/python export_model.py \
 *       models_9x9/157_best_model.pt /tmp/test_model_ts.pt
 *   ./test_nn /tmp/test_model_ts.pt
 */

#include "nn_inference.h"
#include "mcts.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>

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
    if (tests_failed > 0) printf("  (%d FAILED)", tests_failed);
    printf(" ===\n");
}

/* ── Pool (static to avoid stack overflow) ────────────────────────────── */

static NodePool s_pool;

/* ── Tests ────────────────────────────────────────────────────────────── */

static void test_single_batch(NNInference &nn)
{
    printf("\n[single-board inference]\n");

    float planes[3 * NUM_POSITIONS] = {};
    /* Place a current-player stone at the centre (plane 0). */
    planes[NUM_POSITIONS / 2] = 1.0f;

    float value;
    float policy[ACTION_SIZE];
    nn.eval(planes, 1, &value, policy);

    EXPECT(value >= -1.0f && value <= 1.0f,
           "value should be in [-1, 1]");

    float psum = 0.0f;
    int   all_nonneg = 1;
    for (int a = 0; a < ACTION_SIZE; a++) {
        if (policy[a] < 0.0f) all_nonneg = 0;
        psum += policy[a];
    }
    EXPECT(all_nonneg, "all policy entries should be >= 0 (softmax output)");
    EXPECT(fabsf(psum - 1.0f) < 1e-4f,
           "policy should sum to 1.0 after softmax");
}

static void test_batch_of_8(NNInference &nn)
{
    printf("\n[batch inference – 8 boards]\n");

    const int B = 8;
    float planes[B * 3 * NUM_POSITIONS] = {};
    float values[B];
    float policies[B * ACTION_SIZE];

    /* Give each board a different stone pattern. */
    for (int b = 0; b < B; b++)
        planes[b * 3 * NUM_POSITIONS + b * 3] = 1.0f;  /* plane 0, pos b*3 */

    nn.eval(planes, B, values, policies);

    int all_values_valid = 1;
    for (int b = 0; b < B; b++)
        if (values[b] < -1.0f || values[b] > 1.0f) all_values_valid = 0;
    EXPECT(all_values_valid, "all values in batch should be in [-1, 1]");

    int all_policy_sums_ok = 1;
    for (int b = 0; b < B; b++) {
        float s = 0.0f;
        for (int a = 0; a < ACTION_SIZE; a++)
            s += policies[b * ACTION_SIZE + a];
        if (fabsf(s - 1.0f) > 1e-4f) all_policy_sums_ok = 0;
    }
    EXPECT(all_policy_sums_ok, "every policy in batch should sum to 1.0");
}

static void test_empty_board_symmetry(NNInference &nn)
{
    printf("\n[empty board produces uniform-ish policy]\n");

    float planes[3 * NUM_POSITIONS] = {};
    /* Plane 2 = empty: all 1s. */
    for (int i = 0; i < NUM_POSITIONS; i++)
        planes[2 * NUM_POSITIONS + i] = 1.0f;

    float value;
    float policy[ACTION_SIZE];
    nn.eval(planes, 1, &value, policy);

    /* On a blank board, policy should not collapse to a single action. */
    float pmax = 0.0f;
    for (int a = 0; a < ACTION_SIZE; a++)
        if (policy[a] > pmax) pmax = policy[a];

    /* A totally uniform distribution gives 1/82 ≈ 0.012. A trained model
     * on an empty board should not have one action with > 99% probability. */
    EXPECT(pmax < 0.99f,
           "policy on empty board should not be degenerate (one action dominating)");
}

static void test_deterministic(NNInference &nn)
{
    printf("\n[inference is deterministic]\n");

    float planes[3 * NUM_POSITIONS] = {};
    planes[0] = 1.0f;  /* arbitrary pattern */

    float v1, v2;
    float p1[ACTION_SIZE], p2[ACTION_SIZE];
    nn.eval(planes, 1, &v1, p1);
    nn.eval(planes, 1, &v2, p2);

    EXPECT(v1 == v2, "same input → same value both calls");
    int same = 1;
    for (int a = 0; a < ACTION_SIZE; a++)
        if (p1[a] != p2[a]) { same = 0; break; }
    EXPECT(same, "same input → identical policy both calls");
}

static void test_mcts_with_real_nn(NNInference &nn)
{
    printf("\n[MCTS integration with real NN]\n");

    /* Wrap NNInference as a NNEvalFn using a file-scope pointer. */
    static NNInference *g_nn;
    g_nn = &nn;

    NNEvalFn fn = [](const float *p, int bs, float *v, float *pol) {
        g_nn->eval(p, bs, v, pol);
    };

    go_init();
    GoState s = go_initial_state();
    mcts_init_root(&s_pool, &s, 1);
    mcts_simulate(&s_pool, fn, 50, 8, false);

    EXPECT(s_pool.nodes[0].visits == 50,
           "root should have 50 visits after 50 sims");

    float probs[ACTION_SIZE];
    int move = mcts_select_move(&s_pool, 0.0f, probs);
    EXPECT(move >= 0 && move < ACTION_SIZE,
           "selected move should be a valid action index");

    float psum = 0.0f;
    for (int a = 0; a < ACTION_SIZE; a++) psum += probs[a];
    EXPECT(fabsf(psum - 1.0f) < 1e-4f,
           "action probs from MCTS should sum to 1");

    printf("  selected move: %d  (pass=%d)\n", move, PASS_ACTION);
    printf("  nodes used: %d\n", s_pool.next_free);
}

/* ── main ─────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <torchscript_model.pt>\n"
                "\n"
                "Generate the model with:\n"
                "  BOARD_SIZE=9 .venv/bin/python export_model.py "
                "<state_dict.pt> <ts.pt>\n",
                argv[0]);
        return 1;
    }

    printf("=== nn_inference tests ===\n");
    printf("Model: %s\n", argv[1]);

    NNInference nn(argv[1], /*use_cuda=*/false);
    printf("Device: %s\n", nn.on_cuda() ? "cuda" : "cpu");

    go_init();

    test_single_batch(nn);
    test_batch_of_8(nn);
    test_empty_board_symmetry(nn);
    test_deterministic(nn);
    test_mcts_with_real_nn(nn);

    print_summary();
    return (tests_failed > 0) ? 1 : 0;
}
