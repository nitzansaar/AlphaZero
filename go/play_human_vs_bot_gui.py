import os
import numpy as np
from glob import glob
import torch
import tkinter as tk
from tkinter import messagebox
import threading

from config import Config as cfg
from game import Go, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE
from mcts import MonteCarloTreeSearch, Node
from value_policy_function import ValuePolicyNetwork
from model import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load the latest trained model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_dirs = [
        os.path.join(script_dir, cfg.SAVE_MODEL_PATH),
        cfg.SAVE_MODEL_PATH,
    ]

    for model_dir in possible_dirs:
        all_models = glob(os.path.join(model_dir, "*_best_model.pt"))
        if all_models:
            models_with_time = []
            for f in all_models:
                try:
                    mtime = os.path.getmtime(f)
                    models_with_time.append((mtime, f))
                except OSError:
                    continue

            if models_with_time:
                models_with_time.sort(reverse=True)

                for mtime, model_file in models_with_time:
                    try:
                        test_model = NeuralNetwork().to(device)
                        test_state = torch.load(model_file, map_location=device)
                        test_model.load_state_dict(test_state)
                        return model_file
                    except (RuntimeError, FileNotFoundError):
                        continue
                    finally:
                        del test_model
                        if 'test_state' in locals():
                            del test_state
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return None


class GoGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("5x5 Go - Human vs AlphaZero Bot")
        self.root.geometry("600x750")
        self.root.resizable(False, False)
        self.root.configure(bg="#F0F0F0")

        self.human_player = tk.IntVar(value=1)
        self.num_simulations = tk.IntVar(value=800)

        self.game = None
        self.mcts = None
        self.policy_value_network = None

        self.state = None
        self.current_player = 1
        self.game_over = False
        self.is_human_turn = False

        self.canvas = None
        self.status_label = None
        self.info_label = None
        self.heatmap_var = tk.BooleanVar(value=True)
        self._last_mcts_visit_counts = None
        self._last_mcts_chosen_action_index = None

        self.cell_size = 100
        self.board_size = BOARD_SIZE
        self.canvas_size = self.cell_size * self.board_size

        self.color_black = "#000000"
        self.color_white = "#FFFFFF"
        self.color_grid = "#333333"
        self.color_bg = "#DEB887"  # Tan/wood color for Go board
        self.color_highlight = "#90EE90"

        self.show_settings_screen()

    def show_settings_screen(self):
        """Display the initial settings screen."""
        for widget in self.root.winfo_children():
            widget.destroy()

        main_frame = tk.Frame(self.root, bg="#F0F0F0")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        title = tk.Label(main_frame, text="5x5 Go", font=("Arial", 24, "bold"), bg="#F0F0F0")
        title.pack(pady=(0, 10))

        subtitle = tk.Label(main_frame, text="Human vs AlphaZero Bot", font=("Arial", 16), bg="#F0F0F0")
        subtitle.pack(pady=(0, 30))

        settings_frame = tk.Frame(main_frame, bg="#F0F0F0")
        settings_frame.pack(pady=20)

        first_label = tk.Label(settings_frame, text="Who goes first?", font=("Arial", 14, "bold"), bg="#F0F0F0")
        first_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)

        human_first_rb = tk.Radiobutton(settings_frame, text="I go first (Black)", variable=self.human_player,
                                        value=1, font=("Arial", 12), bg="#F0F0F0")
        human_first_rb.grid(row=1, column=0, sticky=tk.W, padx=(20, 0))

        bot_first_rb = tk.Radiobutton(settings_frame, text="Bot goes first (Black)", variable=self.human_player,
                                      value=-1, font=("Arial", 12), bg="#F0F0F0")
        bot_first_rb.grid(row=2, column=0, sticky=tk.W, padx=(20, 0))

        difficulty_label = tk.Label(settings_frame, text="Choose Difficulty", font=("Arial", 14, "bold"), bg="#F0F0F0")
        difficulty_label.grid(row=3, column=0, columnspan=2, pady=(30, 10), sticky=tk.W)

        for i, (text, value) in enumerate([("Easy", 200), ("Medium", 400), ("Hard", 800), ("Expert", 1600)]):
            rb = tk.Radiobutton(settings_frame, text=text, variable=self.num_simulations,
                                value=value, font=("Arial", 12), bg="#F0F0F0")
            rb.grid(row=4+i, column=0, sticky=tk.W, padx=(20, 0))

        start_button = tk.Button(main_frame, text="Start Game", command=self.start_game,
                                 font=("Arial", 14, "bold"), bg="#4CAF50", fg="white",
                                 padx=30, pady=10, cursor="hand2")
        start_button.pack(pady=30)

        quit_button = tk.Button(main_frame, text="Quit", command=self.root.quit,
                                font=("Arial", 12), bg="#f44336", fg="white",
                                padx=20, pady=5, cursor="hand2")
        quit_button.pack()

    def start_game(self):
        """Initialize game components and show game screen."""
        try:
            model_path = load_model()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            return

        if model_path is None:
            messagebox.showerror("Error", "No trained model found!\n\nPlease train a model first.")
            return

        try:
            self.game = Go()
            vpn = ValuePolicyNetwork(model_path, use_compile=False)
            self.policy_value_network = vpn.get_vp
            self.mcts = MonteCarloTreeSearch(self.game, self.policy_value_network)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize game:\n{str(e)}")
            return

        self.show_game_screen()

    def show_game_screen(self):
        """Display the main game screen."""
        for widget in self.root.winfo_children():
            widget.destroy()

        self.state = self.game.state.copy()
        self.current_player = 1
        self.game_over = False
        self.is_human_turn = (self.current_player == self.human_player.get())
        self._last_mcts_visit_counts = None
        self._last_mcts_chosen_action_index = None

        main_frame = tk.Frame(self.root, bg="#F0F0F0")
        main_frame.pack(expand=True, fill=tk.BOTH)

        self.status_label = tk.Label(main_frame, text="", font=("Arial", 16, "bold"),
                                     bg="#90EE90", fg="black", height=2)
        self.status_label.pack(fill=tk.X)

        canvas_frame = tk.Frame(main_frame, bg="#F0F0F0")
        canvas_frame.pack(pady=10)

        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size,
                                bg=self.color_bg, highlightthickness=2, highlightbackground=self.color_grid)
        self.canvas.pack()

        self.draw_grid()
        self.update_board_display()

        self.canvas.bind("<Button-1>", self.handle_board_click)
        self.canvas.bind("<Motion>", self.handle_mouse_motion)

        # Info label for scores
        self.info_label = tk.Label(main_frame, text="", font=("Arial", 12), bg="#F0F0F0")
        self.info_label.pack(pady=5)

        # Pass button
        button_frame = tk.Frame(main_frame, bg="#F0F0F0")
        button_frame.pack(pady=10)

        pass_btn = tk.Button(button_frame, text="Pass", command=self.human_pass,
                             font=("Arial", 12), padx=15, pady=5, cursor="hand2")
        pass_btn.grid(row=0, column=0, padx=5)

        new_game_btn = tk.Button(button_frame, text="New Game", command=self.show_game_screen,
                                 font=("Arial", 12), padx=15, pady=5, cursor="hand2")
        new_game_btn.grid(row=0, column=1, padx=5)

        settings_btn = tk.Button(button_frame, text="Settings", command=self.show_settings_screen,
                                 font=("Arial", 12), padx=15, pady=5, cursor="hand2")
        settings_btn.grid(row=0, column=2, padx=5)

        quit_btn = tk.Button(button_frame, text="Quit", command=self.root.quit,
                             font=("Arial", 12), bg="#f44336", fg="white",
                             padx=15, pady=5, cursor="hand2")
        quit_btn.grid(row=0, column=3, padx=5)

        self.update_status()
        self.update_info()

        if not self.is_human_turn:
            self.root.after(500, self.make_bot_move)

    def draw_grid(self):
        """Draw the Go board grid."""
        margin = self.cell_size // 2
        for i in range(self.board_size):
            # Vertical lines
            x = margin + i * self.cell_size
            self.canvas.create_line(x, margin, x, self.canvas_size - margin,
                                    fill=self.color_grid, width=1)
            # Horizontal lines
            y = margin + i * self.cell_size
            self.canvas.create_line(margin, y, self.canvas_size - margin, y,
                                    fill=self.color_grid, width=1)

        # Star points (for 5x5, center point)
        center = self.board_size // 2
        cx = margin + center * self.cell_size
        cy = margin + center * self.cell_size
        r = 4
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=self.color_grid)

    def update_board_display(self):
        """Update the visual display of the board."""
        self.canvas.delete("piece")
        self.canvas.delete("highlight")
        self.canvas.delete("hover")

        board = self.game.get_board(self.state)
        board_2d = board.reshape(BOARD_SIZE, BOARD_SIZE)
        margin = self.cell_size // 2

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board_2d[row, col] == 1:
                    self.draw_stone(row, col, "black")
                elif board_2d[row, col] == -1:
                    self.draw_stone(row, col, "white")

    def draw_stone(self, row, col, color):
        """Draw a Go stone at the specified position."""
        margin = self.cell_size // 2
        cx = margin + col * self.cell_size
        cy = margin + row * self.cell_size
        radius = self.cell_size // 2 - 5

        fill = self.color_black if color == "black" else self.color_white
        outline = self.color_grid

        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                fill=fill, outline=outline, width=2, tags="piece")

    def handle_board_click(self, event):
        """Handle mouse click on the board."""
        if not self.is_human_turn or self.game_over:
            return

        margin = self.cell_size // 2
        col = round((event.x - margin) / self.cell_size)
        row = round((event.y - margin) / self.cell_size)

        if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
            return

        action_index = row * BOARD_SIZE + col

        valid_moves = self.game.get_valid_moves(self.state, self.current_player)
        if valid_moves[action_index] != 1:
            self.show_invalid_move_feedback(row, col)
            return

        self.make_human_move(action_index)

    def handle_mouse_motion(self, event):
        """Handle mouse motion for hover effects."""
        if not self.is_human_turn or self.game_over:
            return

        self.canvas.delete("hover")

        margin = self.cell_size // 2
        col = round((event.x - margin) / self.cell_size)
        row = round((event.y - margin) / self.cell_size)

        if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
            return

        action_index = row * BOARD_SIZE + col
        valid_moves = self.game.get_valid_moves(self.state, self.current_player)

        if valid_moves[action_index] == 1:
            cx = margin + col * self.cell_size
            cy = margin + row * self.cell_size
            radius = self.cell_size // 2 - 10

            self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                    fill=self.color_highlight, outline="", tags="hover")
            self.canvas.tag_lower("hover")

    def show_invalid_move_feedback(self, row, col):
        """Show visual feedback for invalid move."""
        margin = self.cell_size // 2
        cx = margin + col * self.cell_size
        cy = margin + row * self.cell_size
        radius = self.cell_size // 2 - 5

        rect = self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                       outline="red", width=3, tags="invalid")
        self.root.after(500, lambda: self.canvas.delete("invalid"))

    def human_pass(self):
        """Handle human player passing."""
        if not self.is_human_turn or self.game_over:
            return
        self.make_human_move(PASS_ACTION)

    def make_human_move(self, action_index):
        """Process human move."""
        self.state = self.game.apply_move(self.state, action_index, self.current_player)
        self.update_board_display()
        self.update_info()

        if self.check_game_end():
            return

        self.current_player *= -1
        self.is_human_turn = False
        self.update_status()

        self.root.after(300, self.make_bot_move)

    def make_bot_move(self):
        """Initiate bot move in background thread."""
        if self.game_over:
            return

        self.update_status()
        self.canvas.unbind("<Button-1>")

        thread = threading.Thread(target=self._bot_move_thread)
        thread.daemon = True
        thread.start()

    def _bot_move_thread(self):
        """Run MCTS in background thread."""
        try:
            node = Node(prior_prob=0, player=self.current_player, action_index=None)
            node.set_state(self.state.copy())

            root_node = self.mcts.run_simulation(
                root_node=node,
                num_simulations=self.num_simulations.get(),
                player=self.current_player
            )

            action, _, action_probs = self.mcts.select_move(node=root_node, mode="exploit", temperature=0.1)
            action_index = np.argmax(action)

            visit_counts = np.zeros(ACTION_SIZE)
            for k, v in root_node.children.items():
                visit_counts[k] = v.total_visits_N

            self.root.after(0, lambda: self._apply_bot_move(action_index, visit_counts))

        except Exception as e:
            self.root.after(0, lambda: self.handle_bot_error(str(e)))

    def _apply_bot_move(self, action_index, visit_counts):
        """Apply bot move to the game state."""
        self._last_mcts_visit_counts = visit_counts
        self._last_mcts_chosen_action_index = action_index

        self.state = self.game.apply_move(self.state, action_index, self.current_player)
        self.update_board_display()
        self.update_info()

        if self.check_game_end():
            return

        self.current_player *= -1
        self.is_human_turn = True
        self.update_status()

        self.canvas.bind("<Button-1>", self.handle_board_click)

    def handle_bot_error(self, error_msg):
        """Handle errors from bot move thread."""
        messagebox.showerror("Bot Error", f"An error occurred:\n{error_msg}")
        self.show_settings_screen()

    def check_game_end(self):
        """Check if game has ended."""
        result = self.game.win_or_draw(self.state)

        if result is not None:
            self.game_over = True
            self.canvas.unbind("<Button-1>")
            self.show_game_over_dialog(result)
            return True

        return False

    def show_game_over_dialog(self, result):
        """Show game over dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Game Over")
        dialog.geometry("400x350")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Get scores
        black_score, white_score = self.game.count_territory(self.game.get_board(self.state))
        white_score_with_komi = white_score + 2.5

        if result == 1:
            if self.human_player.get() == 1:
                message = "Congratulations!\nYou won!"
                color = "#4CAF50"
            else:
                message = "Bot (Black) wins!"
                color = "#f44336"
        elif result == -1:
            if self.human_player.get() == -1:
                message = "Congratulations!\nYou won!"
                color = "#4CAF50"
            else:
                message = "Bot (White) wins!"
                color = "#f44336"
        else:
            message = "It's a draw!"
            color = "#9E9E9E"

        result_label = tk.Label(dialog, text=message, font=("Arial", 20, "bold"),
                                bg=color, fg="white", pady=20)
        result_label.pack(fill=tk.X)

        score_label = tk.Label(dialog, text=f"Black: {black_score}  |  White: {white_score_with_komi:.1f} (incl. 2.5 komi)",
                               font=("Arial", 12), pady=10)
        score_label.pack()

        button_frame = tk.Frame(dialog, bg="#F0F0F0")
        button_frame.pack(expand=True, pady=20)

        def play_again():
            dialog.destroy()
            self.show_game_screen()

        def switch_sides():
            dialog.destroy()
            self.human_player.set(self.human_player.get() * -1)
            self.show_game_screen()

        def main_menu():
            dialog.destroy()
            self.show_settings_screen()

        play_again_btn = tk.Button(button_frame, text="Play Again", command=play_again,
                                   font=("Arial", 12), bg="#4CAF50", fg="white",
                                   padx=20, pady=10, cursor="hand2")
        play_again_btn.grid(row=0, column=0, columnspan=2, pady=5, padx=5, sticky=tk.EW)

        switch_btn = tk.Button(button_frame, text="Switch Sides", command=switch_sides,
                               font=("Arial", 12), padx=20, pady=10, cursor="hand2")
        switch_btn.grid(row=1, column=0, pady=5, padx=5, sticky=tk.EW)

        menu_btn = tk.Button(button_frame, text="Main Menu", command=main_menu,
                             font=("Arial", 12), padx=20, pady=10, cursor="hand2")
        menu_btn.grid(row=1, column=1, pady=5, padx=5, sticky=tk.EW)

        quit_btn = tk.Button(button_frame, text="Quit", command=self.root.quit,
                             font=("Arial", 12), bg="#f44336", fg="white",
                             padx=20, pady=10, cursor="hand2")
        quit_btn.grid(row=2, column=0, columnspan=2, pady=5, padx=5, sticky=tk.EW)

    def update_status(self):
        """Update the status label."""
        if self.game_over:
            return

        if self.is_human_turn:
            player_color = "Black" if self.human_player.get() == 1 else "White"
            self.status_label.config(text=f"Your turn ({player_color})", bg="#90EE90")
        else:
            player_color = "Black" if self.current_player == 1 else "White"
            self.status_label.config(text=f"Bot thinking... ({player_color})", bg="#FFFF99")

    def update_info(self):
        """Update the info label with current scores."""
        if self.info_label is None:
            return
        board = self.game.get_board(self.state)
        black_score, white_score = self.game.count_territory(board)
        ko = self.game.get_ko_point(self.state)
        passes = self.game.get_consecutive_passes(self.state)
        ko_str = f"Ko: {ko}" if ko >= 0 else "Ko: None"
        self.info_label.config(text=f"Black: {black_score} | White: {white_score + 2.5:.1f} | {ko_str} | Passes: {passes}")

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = GoGUI()
    app.run()


if __name__ == "__main__":
    main()
