/*
 * tests/test_go_engine.c
 *
 * Compile:
 *   gcc -O2 -Wall -I.. -o test_go_engine \
 *       ../go_engine.c test_go_engine.c && ./test_go_engine
 */

#include "go_engine.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ── Minimal test harness ─────────────────────────────────────────────── */

static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define EXPECT(cond, msg) do {                               \
    tests_run++;                                             \
    if (cond) {                                              \
        tests_passed++;                                      \
    } else {                                                 \
        tests_failed++;                                      \
        printf("  FAIL [line %d]: %s\n", __LINE__, (msg));  \
    }                                                        \
} while (0)

static void print_summary(void)
{
    printf("\n=== Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0)
        printf("  (%d FAILED)", tests_failed);
    printf(" ===\n");
}

/* ── Helpers ──────────────────────────────────────────────────────────── */

/* idx of (row, col) for the current BOARD_SIZE */
static int rc(int row, int col) { return row * BOARD_SIZE + col; }

/* ── Test suites ──────────────────────────────────────────────────────── */

static void test_initial_state(void)
{
    printf("\n[initial state]\n");
    GoState s = go_initial_state();

    EXPECT(s.ko_point == -1,           "ko_point should start at -1");
    EXPECT(s.consecutive_passes == 0,  "passes should start at 0");

    int all_empty = 1;
    for (int i = 0; i < NUM_POSITIONS; i++)
        if (s.board[i] != 0) { all_empty = 0; break; }
    EXPECT(all_empty, "board should be all empty");
}

static void test_neighbor_table(void)
{
    printf("\n[neighbor table]\n");

    /* Corner: (0,0) has exactly 2 neighbours */
    int nb[4];
    int count = go_get_neighbors(rc(0,0), nb);
    EXPECT(count == 2, "corner (0,0) should have 2 neighbours");

    /* Edge: (0, BOARD_SIZE/2) has 3 neighbours */
    count = go_get_neighbors(rc(0, BOARD_SIZE/2), nb);
    EXPECT(count == 3, "top-edge cell should have 3 neighbours");

    /* Interior: (1,1) has 4 neighbours */
    count = go_get_neighbors(rc(1,1), nb);
    EXPECT(count == 4, "interior cell (1,1) should have 4 neighbours");

    /* Centre cell should also have 4 neighbours */
    int centre = rc(BOARD_SIZE/2, BOARD_SIZE/2);
    count = go_get_neighbors(centre, nb);
    EXPECT(count == 4, "centre should have 4 neighbours");
}

static void test_find_group_single_stone(void)
{
    printf("\n[find_group – single stone]\n");
    GoState s = go_initial_state();

    /* Place a single black stone at centre. */
    int centre = rc(BOARD_SIZE/2, BOARD_SIZE/2);
    s.board[centre] = 1;

    int group[NUM_POSITIONS], gs;
    int liberties = go_find_group(s.board, centre, group, &gs);

    EXPECT(gs       == 1,        "single stone: group size 1");
    EXPECT(liberties == 4,       "single stone at centre: 4 liberties");
    EXPECT(group[0] == centre,   "group contains the stone itself");
}

static void test_find_group_corner_stone(void)
{
    printf("\n[find_group – corner stone]\n");
    GoState s = go_initial_state();
    s.board[rc(0,0)] = 1;

    int group[NUM_POSITIONS], gs;
    int liberties = go_find_group(s.board, rc(0,0), group, &gs);

    EXPECT(gs       == 1, "corner stone: group size 1");
    EXPECT(liberties == 2, "corner stone: 2 liberties");
}

static void test_find_group_connected(void)
{
    printf("\n[find_group – connected group]\n");
    GoState s = go_initial_state();

    /*
     * 3 black stones in a row: (2,1) (2,2) (2,3)
     * All cells are interior on both 5x5 and 9x9, so the liberty count
     * is the same regardless of board size.
     */
    s.board[rc(2,1)] = 1;
    s.board[rc(2,2)] = 1;
    s.board[rc(2,3)] = 1;

    int group[NUM_POSITIONS], gs;
    int liberties = go_find_group(s.board, rc(2,2), group, &gs);

    EXPECT(gs == 3, "three-in-a-row: group size 3");
    /* (2,1) contributes (1,1),(3,1),(2,0); (2,2) adds (1,2),(3,2);
       (2,3) adds (1,3),(3,3),(2,4) — total 8 unique liberties. */
    EXPECT(liberties == 8, "three-in-a-row: 8 liberties");
}

static void test_find_group_empty(void)
{
    printf("\n[find_group – empty cell]\n");
    GoState s = go_initial_state();

    int group[NUM_POSITIONS], gs;
    int liberties = go_find_group(s.board, rc(3,3), group, &gs);

    EXPECT(gs == 0,       "empty cell: group size 0");
    EXPECT(liberties == 0, "empty cell: 0 liberties");
}

static void test_capture_single_stone(void)
{
    printf("\n[capture – single stone surrounded]\n");
    GoState s = go_initial_state();

    /*
     *  . X .
     *  X O X   <- white stone at (1,1) surrounded on all four sides
     *  . X .
     */
    int centre = rc(1,1);
    s.board[rc(0,1)] = 1;
    s.board[rc(1,0)] = 1;
    s.board[rc(1,2)] = 1;
    s.board[rc(2,1)] = 1;
    s.board[centre]  = -1;   /* white */

    /* Capture white stones (player=1 → opponent=-1). */
    int captured = go_capture_dead_stones(s.board, 1);

    EXPECT(captured == 1,            "should capture 1 stone");
    EXPECT(s.board[centre] == 0,     "captured position should be empty");
    EXPECT(s.board[rc(0,1)] == 1,    "black stones should remain");
}

static void test_capture_two_stone_group(void)
{
    printf("\n[capture – two-stone group]\n");
    GoState s = go_initial_state();

    /*
     *  . X X .
     *  X O O X   <- white pair at (1,1) and (1,2)
     *  . X X .
     */
    s.board[rc(0,1)] = 1;  s.board[rc(0,2)] = 1;
    s.board[rc(1,0)] = 1;  s.board[rc(1,3)] = 1;
    s.board[rc(2,1)] = 1;  s.board[rc(2,2)] = 1;
    s.board[rc(1,1)] = -1; s.board[rc(1,2)] = -1;

    int captured = go_capture_dead_stones(s.board, 1);

    EXPECT(captured == 2,         "should capture 2 stones");
    EXPECT(s.board[rc(1,1)] == 0, "(1,1) should be empty");
    EXPECT(s.board[rc(1,2)] == 0, "(1,2) should be empty");
}

static void test_no_capture_with_liberty(void)
{
    printf("\n[capture – stone with liberty not captured]\n");
    GoState s = go_initial_state();

    /* White at (1,1) — surrounded on only 3 sides. */
    s.board[rc(0,1)] = 1;
    s.board[rc(1,0)] = 1;
    s.board[rc(2,1)] = 1;
    s.board[rc(1,1)] = -1;

    int captured = go_capture_dead_stones(s.board, 1);

    EXPECT(captured == 0,         "should not capture: stone has liberty");
    EXPECT(s.board[rc(1,1)] == -1, "white stone should remain");
}

static void test_is_suicide(void)
{
    printf("\n[suicide]\n");
    GoState s = go_initial_state();

    /*
     * Place black stones around (1,1), leaving it empty.
     * Playing white at (1,1) with no liberties and no captures → suicide.
     */
    s.board[rc(0,1)] = 1;
    s.board[rc(1,0)] = 1;
    s.board[rc(1,2)] = 1;
    s.board[rc(2,1)] = 1;

    EXPECT(go_is_suicide(s.board, rc(1,1), -1) == 1,
           "surrounded empty point should be suicide for white");

    /* Playing black at the same spot is not suicide (has adjacent empties). */
    GoState s2 = go_initial_state();
    EXPECT(go_is_suicide(s2.board, rc(1,1), 1) == 0,
           "open board: playing is never suicide");
}

static void test_suicide_that_captures(void)
{
    printf("\n[suicide – capture exemption]\n");

    /*
     * Each white stone's ONLY liberty is (1,1).  All other neighbours of
     * each white stone are black.  Black plays at (1,1), which:
     *   - captures all four surrounding white stones (captured >= 1)
     *   - is therefore NOT suicide, even though black's new stone would
     *     have no liberties if the captures weren't performed first.
     *
     *   col: 0  1  2
     *   row0: X  O  X
     *   row1: O  .  O   <- (1,1) is the target
     *   row2: X  O  X
     *   row3: .  X  .   <- blocks (2,1)'s southern liberty
     */
    GoState s = go_initial_state();
    /* Corner/edge black stones that block every external liberty. */
    s.board[rc(0,0)] =  1;  s.board[rc(0,2)] =  1;
    s.board[rc(2,0)] =  1;  s.board[rc(2,2)] =  1;
    s.board[rc(3,1)] =  1;
    /* White ring — each stone's only liberty is (1,1). */
    s.board[rc(0,1)] = -1;
    s.board[rc(1,0)] = -1;
    s.board[rc(1,2)] = -1;
    s.board[rc(2,1)] = -1;

    EXPECT(go_is_suicide(s.board, rc(1,1), 1) == 0,
           "move that captures all surrounding opponents is not suicide");
}

static void test_ko_detection(void)
{
    printf("\n[ko detection]\n");

    /*
     * Ko setup requirements:
     *   1. exactly one opponent stone captured
     *   2. the placing stone is a SINGLETON group (not connected to any
     *      same-colour stone)
     *   3. the singleton has exactly ONE liberty (the just-vacated cell)
     *
     * Layout (top-edge makes white at (0,1) have only 3 neighbours):
     *
     *   col: 0  1  2  ...
     *   row0: X  O  X  .  .  .  .  .  .
     *   row1: O  .  O  .  .  .  .  .  .
     *   row2: .  O  .  .  .  .  .  .  .
     *
     *   White at (0,1): neighbours (0,0)=black, (0,2)=black, (1,1)=empty.
     *   One liberty at (1,1).
     *
     *   Black plays (1,1):
     *     - captures white at (0,1) only (captured=1)
     *     - black at (1,1) neighbours: (0,1)=empty, (1,0)=white, (1,2)=white, (2,1)=white
     *     - group_size=1, liberties=1  →  ko_point = (0,1)
     *
     *   White at (1,0),(1,2),(2,1) still have external liberties so they
     *   are NOT captured.
     */
    GoState s = go_initial_state();
    s.board[rc(0,0)] =  1;   /* black blocks (0,1)'s left liberty  */
    s.board[rc(0,2)] =  1;   /* black blocks (0,1)'s right liberty */
    s.board[rc(0,1)] = -1;   /* white — only liberty is (1,1)      */
    s.board[rc(1,0)] = -1;   /* white neighbour with external lib  */
    s.board[rc(1,2)] = -1;   /* white neighbour with external lib  */
    s.board[rc(2,1)] = -1;   /* white neighbour with external lib  */

    GoState ns = go_apply_move(&s, rc(1,1), 1);   /* black captures */

    EXPECT(ns.board[rc(0,1)] == 0,   "captured white stone removed");
    EXPECT(ns.board[rc(1,1)] == 1,   "black stone placed at (1,1)");
    EXPECT(ns.ko_point == rc(0,1),   "ko point set to the captured cell (0,1)");

    /* White must not immediately recapture the ko. */
    EXPECT(go_is_valid_move(&ns, rc(0,1), -1) == 0,
           "ko point should be invalid for white to play immediately");
}

static void test_no_ko_on_multi_liberty_capture(void)
{
    printf("\n[apply_move – single capture with multiple liberties is not ko]\n");

    /*
     * White at (0,0) has two neighbours: (0,1) and (1,0).
     * Black at (1,0) fills the southern liberty.
     * Black plays (0,1), capturing white at (0,0).
     *
     * After capture, black at (0,1) has 3 liberties: (0,0), (0,2), (1,1).
     * Ko requires group_size == 1 AND liberties == 1; here liberties == 3,
     * so ko_point must remain -1.
     *
     *   col: 0  1  2
     *   row0: O  .  .    <- white, captured by black at (0,1)
     *   row1: X  .  .    <- black, blocks white's south liberty
     */
    GoState s = go_initial_state();
    s.board[rc(0,0)] = -1;   /* white */
    s.board[rc(1,0)] =  1;   /* black — blocks white's south liberty */

    GoState ns = go_apply_move(&s, rc(0,1), 1);   /* black captures */

    EXPECT(ns.board[rc(0,0)] == 0,  "white stone captured");
    EXPECT(ns.board[rc(0,1)] == 1,  "black stone placed");
    EXPECT(ns.ko_point == -1,
           "no ko: capturing stone has 3 liberties, not 1");
}

static void test_ko_cleared_after_pass_allows_play(void)
{
    printf("\n[ko – formerly forbidden cell is legal after pass]\n");

    /* Reproduce the ko setup from test_ko_detection. */
    GoState s = go_initial_state();
    s.board[rc(0,0)] =  1;
    s.board[rc(0,2)] =  1;
    s.board[rc(0,1)] = -1;
    s.board[rc(1,0)] = -1;
    s.board[rc(1,2)] = -1;
    s.board[rc(2,1)] = -1;

    GoState after_capture = go_apply_move(&s, rc(1,1), 1);
    EXPECT(after_capture.ko_point == rc(0,1), "ko set at (0,1)");
    EXPECT(go_is_valid_move(&after_capture, rc(0,1), -1) == 0,
           "white cannot recapture ko immediately");

    /* Black passes — ko is cleared. */
    GoState after_pass = go_apply_move(&after_capture, PASS_ACTION, 1);
    EXPECT(after_pass.ko_point == -1, "pass clears ko_point");

    /* White can now legally play the formerly-forbidden cell. */
    EXPECT(go_is_valid_move(&after_pass, rc(0,1), -1) == 1,
           "after pass, formerly ko'd cell is now legal for white");
}

static void test_valid_moves_on_empty_board(void)
{
    printf("\n[valid moves – empty board]\n");
    GoState s = go_initial_state();

    float mask[ACTION_SIZE];
    go_get_valid_moves(&s, 1, mask);

    int all_valid = 1;
    for (int i = 0; i < ACTION_SIZE; i++)
        if (mask[i] != 1.0f) { all_valid = 0; break; }

    EXPECT(all_valid, "all moves valid on empty board");
}

static void test_occupied_square_invalid(void)
{
    printf("\n[valid moves – occupied square]\n");
    GoState s = go_initial_state();
    s.board[rc(3,3)] = 1;

    EXPECT(go_is_valid_move(&s, rc(3,3), -1) == 0,
           "cannot play on occupied square");
    EXPECT(go_is_valid_move(&s, rc(3,3),  1) == 0,
           "cannot play on own stone either");
}

static void test_apply_move_pass(void)
{
    printf("\n[apply_move – pass]\n");
    GoState s = go_initial_state();

    GoState s1 = go_apply_move(&s,  PASS_ACTION, 1);
    EXPECT(s1.consecutive_passes == 1, "one pass → passes=1");
    EXPECT(s1.ko_point == -1,          "ko cleared on pass");

    GoState s2 = go_apply_move(&s1, PASS_ACTION, -1);
    EXPECT(s2.consecutive_passes == 2, "two passes → passes=2");
    EXPECT(go_game_ended(&s2) == 1,    "game ended after two passes");
}

static void test_pass_resets_ko(void)
{
    printf("\n[apply_move – pass clears ko]\n");
    GoState s = go_initial_state();
    s.ko_point = rc(3,3);  /* simulate an active ko */

    GoState ns = go_apply_move(&s, PASS_ACTION, 1);
    EXPECT(ns.ko_point == -1, "pass should clear the ko point");
}

static void test_stone_placement_resets_passes(void)
{
    printf("\n[apply_move – stone resets pass count]\n");
    GoState s = go_initial_state();
    s.consecutive_passes = 1;

    GoState ns = go_apply_move(&s, rc(4,4), 1);
    EXPECT(ns.consecutive_passes == 0, "placing a stone resets pass count");
}

static void test_game_ended(void)
{
    printf("\n[game_ended]\n");
    GoState s = go_initial_state();
    EXPECT(go_game_ended(&s) == 0,  "game not ended at start");

    s.consecutive_passes = 1;
    EXPECT(go_game_ended(&s) == 0,  "one pass: not ended");

    s.consecutive_passes = 2;
    EXPECT(go_game_ended(&s) == 1,  "two passes: ended");
}

static void test_get_winner_black(void)
{
    printf("\n[get_winner – black wins]\n");
    GoState s = go_initial_state();
    s.consecutive_passes = 2;

    /* Black controls left column, white controls nothing. */
    for (int row = 0; row < BOARD_SIZE; row++)
        s.board[rc(row, 0)] = 1;

    int winner = go_get_winner(&s, 1);

    /* Black has BOARD_SIZE stones + territory; white has 0 + komi.
       Black should win on any board since BOARD_SIZE > komi. */
    EXPECT(winner == 1, "black should win when controlling a column");
}

static void test_get_winner_draw_unlikely(void)
{
    printf("\n[get_winner – balanced position]\n");
    GoState s = go_initial_state();
    s.consecutive_passes = 2;

    /* Fill board symmetrically so black and white have equal stone counts. */
    for (int i = 0; i < NUM_POSITIONS; i++)
        s.board[i] = (i % 2 == 0) ? 1 : -1;

    int winner = go_get_winner(&s, 1);
    /* With komi white gets a bonus, so white should win or draw. */
    EXPECT(winner == -1 || winner == 0,
           "with komi white should win or draw on balanced board");
}

static void test_next_state_canonical(void)
{
    printf("\n[next_state_canonical]\n");
    GoState s = go_initial_state();

    int move = rc(4,4);
    GoState ns = go_next_state_canonical(&s, move);

    /* After a canonical move by "current player" (==1), the board is
       flipped.  The just-placed stone was 1 → becomes -1 in the
       canonical view of the NEXT player. */
    EXPECT(ns.board[move] == -1,
           "placed stone should appear as -1 from next player's view");

    /* All other cells should still be empty (0 negated = 0). */
    int all_others_empty = 1;
    for (int i = 0; i < NUM_POSITIONS; i++)
        if (i != move && ns.board[i] != 0) { all_others_empty = 0; break; }
    EXPECT(all_others_empty, "all other cells still empty after canonical move");
}

static void test_board_to_planes(void)
{
    printf("\n[board_to_planes]\n");
    GoState s = go_initial_state();

    int black_pos = rc(2,2);
    int white_pos = rc(3,3);
    s.board[black_pos] =  1;
    s.board[white_pos] = -1;

    float planes[3 * NUM_POSITIONS];

    /* From black's perspective (player=1, absolute_player=1). */
    go_board_to_planes(&s, 1, 1, planes);

    float *p0 = planes;
    float *p1 = planes + NUM_POSITIONS;
    float *p2 = planes + 2 * NUM_POSITIONS;

    EXPECT(p0[black_pos] == 1.0f,   "plane0: black stone as current player");
    EXPECT(p0[white_pos] == 0.0f,   "plane0: white stone not current player");
    EXPECT(p1[white_pos] == 1.0f,   "plane1: white stone as opponent");
    EXPECT(p1[black_pos] == 0.0f,   "plane1: black stone not opponent");
    EXPECT(p2[rc(0,0)]   == 1.0f,   "plane2: color-to-play=1 (Black to move)");
    EXPECT(p2[black_pos] == 1.0f,   "plane2: color-to-play=1 everywhere");

    /* From white's perspective (player=-1, absolute_player=-1). */
    go_board_to_planes(&s, -1, -1, planes);
    p0 = planes; p1 = planes + NUM_POSITIONS; p2 = planes + 2*NUM_POSITIONS;

    EXPECT(p0[white_pos] == 1.0f,   "plane0: white stone as current player");
    EXPECT(p1[black_pos] == 1.0f,   "plane1: black stone as opponent (from white's view)");
    EXPECT(p2[rc(0,0)]   == 0.0f,   "plane2: color-to-play=0 (White to move)");
}

static void test_full_game_sequence(void)
{
    printf("\n[full game – two passes end game]\n");
    GoState s = go_initial_state();

    /* Play a short sequence then two passes. */
    s = go_apply_move(&s, rc(2,2), 1);
    s = go_apply_move(&s, rc(6,6), -1);
    EXPECT(go_game_ended(&s) == 0, "game still in progress");

    s = go_apply_move(&s, PASS_ACTION, 1);
    s = go_apply_move(&s, PASS_ACTION, -1);
    EXPECT(go_game_ended(&s) == 1, "game ended after two passes");
}

/* ── main ─────────────────────────────────────────────────────────────── */

int main(void)
{
    go_init();

    printf("=== go_engine tests  (BOARD_SIZE=%d) ===\n", BOARD_SIZE);

    test_initial_state();
    test_neighbor_table();
    test_find_group_single_stone();
    test_find_group_corner_stone();
    test_find_group_connected();
    test_find_group_empty();
    test_capture_single_stone();
    test_capture_two_stone_group();
    test_no_capture_with_liberty();
    test_is_suicide();
    test_suicide_that_captures();
    test_ko_detection();
    test_no_ko_on_multi_liberty_capture();
    test_ko_cleared_after_pass_allows_play();
    test_valid_moves_on_empty_board();
    test_occupied_square_invalid();
    test_apply_move_pass();
    test_pass_resets_ko();
    test_stone_placement_resets_passes();
    test_game_ended();
    test_get_winner_black();
    test_get_winner_draw_unlikely();
    test_next_state_canonical();
    test_board_to_planes();
    test_full_game_sequence();

    print_summary();

    return (tests_failed > 0) ? 1 : 0;
}
