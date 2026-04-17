#pragma once

/*
 * go_engine.h — Pure C Go game logic for the C++ selfplay binary.
 *
 * Board size is fixed at 9x9 to match the Go implementation in game.py.
 *
 * State layout mirrors game.py:
 *   board[0..NUM_POSITIONS-1]  0=empty, 1=black, -1=white
 *   ko_point                   -1 if none, else the forbidden position index
 *   consecutive_passes         0, 1, or 2
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Board size (9x9 only) ────────────────────────────────────────────── */

#define BOARD_SIZE    9
#define NUM_POSITIONS 81        /* 9 * 9 */
#define PASS_ACTION   81
#define ACTION_SIZE   82        /* 81 positions + 1 pass */
#define KOMI          6.0f    /* standard 9x9 komi */

/* ── Game state ───────────────────────────────────────────────────────── */

typedef struct {
    int8_t board[NUM_POSITIONS]; /* 0=empty  1=black  -1=white */
    int    ko_point;             /* -1 if no ko, else position index */
    int    consecutive_passes;   /* 0, 1, or 2 */
} GoState;

/* ── Initialisation ───────────────────────────────────────────────────── */

/* Must be called once before any other function. */
void go_init(void);

/* Return a blank starting state (empty board, ko=-1, passes=0). */
GoState go_initial_state(void);

/* ── Neighbor table ───────────────────────────────────────────────────── */

/* Fill neighbors_out[0..count-1] with adjacent indices; return count. */
int go_get_neighbors(int idx, int neighbors_out[4]);

/* ── Group / liberty finding ──────────────────────────────────────────── */

/*
 * Find the group connected to board[idx].
 * group_out must have room for NUM_POSITIONS ints.
 * Sets *group_size_out to the number of stones in the group.
 * Returns the number of liberties.
 */
int go_find_group(const int8_t *board, int idx,
                  int *group_out, int *group_size_out);

/* ── Board manipulation ───────────────────────────────────────────────── */

/*
 * Remove all opponent groups with zero liberties.
 * Modifies board in place.  Returns number of stones captured.
 */
int go_capture_dead_stones(int8_t *board, int player);

/* Return 1 if placing player's stone at idx would be suicide. */
int go_is_suicide(const int8_t *board, int idx, int player);

/* Return 1 if action == ko_point (ko rule violation). */
int go_would_be_ko(int action, int ko_point);

/* Return 1 if action is a legal move for player in state. */
int go_is_valid_move(const GoState *state, int action, int player);

/*
 * Fill action_mask[ACTION_SIZE] with 1.0f for each legal move, 0.0f
 * otherwise.  Pass (action == PASS_ACTION) is always legal.
 */
void go_get_valid_moves(const GoState *state, int player, float *action_mask);

/*
 * Apply action for player and return the resulting state.
 * The input state is not modified.
 * In canonical form the current player is always +1, so callers that
 * maintain canonical form should pass player=1.
 */
GoState go_apply_move(const GoState *state, int action, int player);

/*
 * Apply action in canonical form and return the state from the NEXT
 * player's perspective (i.e. flip the board).
 * Mirrors game.py::get_next_state_from_next_player_prespective.
 */
GoState go_next_state_canonical(const GoState *state, int action);

/* ── Scoring / game end ───────────────────────────────────────────────── */

/*
 * Area scoring (Chinese rules).
 * board must be in absolute form (1=black, -1=white).
 */
void go_count_territory(const int8_t *board,
                        int *black_score_out, int *white_score_out);

/* Return 1 if the game has ended (two consecutive passes). */
int go_game_ended(const GoState *state);

/*
 * Determine the winner using area scoring + komi.
 * perspective: 1  → state is in absolute form (black=1, white=-1)
 *             -1  → state is in canonical form (current player=1); board
 *                   is negated before scoring.
 * Returns  1 (black wins), -1 (white wins), 0 (draw).
 */
int go_get_winner(const GoState *state, int perspective);

/* ── Neural-network input ─────────────────────────────────────────────── */

/*
 * Convert state to the 3-plane float representation.
 * planes_out must hold 3 * NUM_POSITIONS floats, laid out as:
 *   [0 .. NUM_POSITIONS-1]        plane 0: current-player stones
 *   [NUM_POSITIONS .. 2N-1]       plane 1: opponent stones
 *   [2*NUM_POSITIONS .. 3N-1]     plane 2: color-to-play
 *                                           1.0 if absolute_player == +1 (Black)
 *                                           0.0 if absolute_player == -1 (White)
 * player:          canonical perspective (pass 1 for canonical-form states)
 * absolute_player: +1 = Black to move, -1 = White to move
 */
void go_board_to_planes(const GoState *state, int player, int absolute_player,
                        float *planes_out);

/*
 * Convert state to the AlphaZero 17-plane float representation used by the NN.
 * planes_out must hold 17 * NUM_POSITIONS floats.
 *
 * Layout (matches board_to_canonical_17 in game.py and Python training):
 *   Planes  0-7 : current-player stones (plane 0 = current board;
 *                 planes 1-7 = history boards, zeroed — no history available)
 *   Planes  8-15: opponent stones        (plane 8 = current board;
 *                 planes 9-15 = history, zeroed)
 *   Plane  16   : color-to-play — 1.0 if absolute_player == +1 (Black),
 *                                  0.0 if absolute_player == -1 (White)
 *
 * player:          canonical perspective (pass 1 for canonical-form states)
 * absolute_player: +1 = Black to move, -1 = White to move
 */
void go_board_to_planes_17(const GoState *state, int player, int absolute_player,
                            float *planes_out);

/* ── Debug ────────────────────────────────────────────────────────────── */

/* Print the board to stdout. */
void go_render(const GoState *state);

#ifdef __cplusplus
}
#endif
