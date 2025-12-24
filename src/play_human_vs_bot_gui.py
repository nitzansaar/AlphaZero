import os
import numpy as np
from glob import glob
import torch
import tkinter as tk
from tkinter import messagebox
import threading

from config import Config as cfg
from game import TicTacToe
from mcts import MonteCarloTreeSearch, Node
from value_policy_function import ValuePolicyNetwork
from model import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """
    Load the latest trained model.
    Returns the model path or None if no model found.
    """
    all_models = glob(os.path.join("src/output_tictac/models", "*_best_model.pt"))
    model_path = None

    if all_models:
        # Get modification time for each model
        models_with_time = []
        for f in all_models:
            try:
                mtime = os.path.getmtime(f)
                models_with_time.append((mtime, f))
            except OSError:
                continue

        if models_with_time:
            # Sort by modification time (most recent first)
            models_with_time.sort(reverse=True)

            # Try loading models starting from most recent until one works
            for mtime, model_file in models_with_time:
                try:
                    # Quick test: try to load state dict to check architecture
                    test_model = NeuralNetwork().to(device)
                    test_state = torch.load(model_file, map_location=device)
                    test_model.load_state_dict(test_state)
                    # If we get here, architecture matches!
                    model_path = model_file
                    break
                except (RuntimeError, FileNotFoundError) as e:
                    # Architecture mismatch or file error, try next
                    continue
                finally:
                    # Clean up test model
                    del test_model
                    if 'test_state' in locals():
                        del test_state
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # If no compatible model found, fall back to highest number
            if model_path is None:
                files_with_numbers = []
                for f in all_models:
                    basename = os.path.basename(f)
                    if "_best_model.pt" in basename:
                        try:
                            num = int(basename.split("_")[0])
                            files_with_numbers.append((num, f))
                        except ValueError:
                            continue

                if files_with_numbers:
                    latest_num, model_path = max(files_with_numbers, key=lambda x: x[0])

    if model_path and os.path.exists(model_path):
        return model_path

    return None


class TicTacToeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("9x9 Tic-Tac-Toe - Human vs AlphaZero Bot")
        self.root.geometry("800x900")
        self.root.resizable(False, False)
        self.root.configure(bg="#F0F0F0")

        # Game settings (to be set in settings screen)
        self.human_player = tk.IntVar(value=1)  # 1 or -1
        self.num_simulations = tk.IntVar(value=800)  # Default to Hard

        # Game components (initialized after model loads)
        self.game = None
        self.mcts = None
        self.policy_value_network = None

        # Game state
        self.state = None
        self.current_player = 1
        self.game_over = False
        self.is_human_turn = False

        # GUI components
        self.canvas = None
        self.status_label = None
        self.info_text = None

        # Visual settings
        self.cell_size = 80
        self.board_size = 9
        self.canvas_size = self.cell_size * self.board_size

        # Colors
        self.color_x = "#FF0000"  # Red for X
        self.color_o = "#0000FF"  # Blue for O
        self.color_grid = "#333333"  # Dark gray
        self.color_bg = "#FFFFFF"  # White background
        self.color_highlight = "#FFFFCC"  # Light yellow for hover

        # Show settings screen first
        self.show_settings_screen()

    def show_settings_screen(self):
        """Display the initial settings screen."""
        # Clear all widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create main frame
        main_frame = tk.Frame(self.root, bg="#F0F0F0")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # Title
        title = tk.Label(
            main_frame,
            text="9x9 Tic-Tac-Toe",
            font=("Arial", 24, "bold"),
            bg="#F0F0F0"
        )
        title.pack(pady=(0, 10))

        subtitle = tk.Label(
            main_frame,
            text="Human vs AlphaZero Bot",
            font=("Arial", 16),
            bg="#F0F0F0"
        )
        subtitle.pack(pady=(0, 30))

        # Settings frame
        settings_frame = tk.Frame(main_frame, bg="#F0F0F0")
        settings_frame.pack(pady=20)

        # Who goes first section
        first_label = tk.Label(
            settings_frame,
            text="Who goes first?",
            font=("Arial", 14, "bold"),
            bg="#F0F0F0"
        )
        first_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)

        human_first_rb = tk.Radiobutton(
            settings_frame,
            text="I go first (X)",
            variable=self.human_player,
            value=1,
            font=("Arial", 12),
            bg="#F0F0F0"
        )
        human_first_rb.grid(row=1, column=0, sticky=tk.W, padx=(20, 0))

        bot_first_rb = tk.Radiobutton(
            settings_frame,
            text="Bot goes first (O goes second)",
            variable=self.human_player,
            value=-1,
            font=("Arial", 12),
            bg="#F0F0F0"
        )
        bot_first_rb.grid(row=2, column=0, sticky=tk.W, padx=(20, 0))

        # Difficulty section
        difficulty_label = tk.Label(
            settings_frame,
            text="Choose Difficulty",
            font=("Arial", 14, "bold"),
            bg="#F0F0F0"
        )
        difficulty_label.grid(row=3, column=0, columnspan=2, pady=(30, 10), sticky=tk.W)

        easy_rb = tk.Radiobutton(
            settings_frame,
            text="Easy (200 simulations)",
            variable=self.num_simulations,
            value=200,
            font=("Arial", 12),
            bg="#F0F0F0"
        )
        easy_rb.grid(row=4, column=0, sticky=tk.W, padx=(20, 0))

        medium_rb = tk.Radiobutton(
            settings_frame,
            text="Medium (400 simulations)",
            variable=self.num_simulations,
            value=400,
            font=("Arial", 12),
            bg="#F0F0F0"
        )
        medium_rb.grid(row=5, column=0, sticky=tk.W, padx=(20, 0))

        hard_rb = tk.Radiobutton(
            settings_frame,
            text="Hard (800 simulations)",
            variable=self.num_simulations,
            value=800,
            font=("Arial", 12),
            bg="#F0F0F0"
        )
        hard_rb.grid(row=6, column=0, sticky=tk.W, padx=(20, 0))

        expert_rb = tk.Radiobutton(
            settings_frame,
            text="Expert (1600 simulations)",
            variable=self.num_simulations,
            value=1600,
            font=("Arial", 12),
            bg="#F0F0F0"
        )
        expert_rb.grid(row=7, column=0, sticky=tk.W, padx=(20, 0))

        # Start button
        start_button = tk.Button(
            main_frame,
            text="Start Game",
            command=self.start_game,
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=30,
            pady=10,
            cursor="hand2"
        )
        start_button.pack(pady=30)

        # Quit button
        quit_button = tk.Button(
            main_frame,
            text="Quit",
            command=self.root.quit,
            font=("Arial", 12),
            bg="#f44336",
            fg="white",
            padx=20,
            pady=5,
            cursor="hand2"
        )
        quit_button.pack()

    def start_game(self):
        """Initialize game components and show game screen."""
        print("Start Game button clicked")

        # Load model
        try:
            model_path = load_model()
            print(f"Model path: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            return

        if model_path is None:
            print("No model found!")
            messagebox.showerror(
                "Error",
                "No trained model found!\n\nPlease train a model first using: ./train.sh\n\n" +
                f"Looking in: src/output_tictac/models"
            )
            return

        # Initialize game components
        try:
            print("Initializing game components...")
            self.game = TicTacToe()
            print("TicTacToe created")
            vpn = ValuePolicyNetwork(model_path, use_compile=False)
            print("ValuePolicyNetwork created")
            self.policy_value_network = vpn.get_vp
            print("Policy value function set")
            self.mcts = MonteCarloTreeSearch(self.game, self.policy_value_network)
            print("MCTS created")
        except Exception as e:
            print(f"Exception during initialization: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to initialize game:\n{str(e)}")
            return

        # Show game screen
        print("Showing game screen...")
        self.show_game_screen()
        print("Game screen shown")

    def show_game_screen(self):
        """Display the main game screen."""
        # Clear all widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Reset game state
        self.state = np.zeros(cfg.ACTION_SIZE)
        self.current_player = 1  # X always starts
        self.game_over = False
        self.is_human_turn = (self.current_player == self.human_player.get())

        # Create main container
        main_frame = tk.Frame(self.root, bg="#F0F0F0")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Status bar
        self.status_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 16, "bold"),
            bg="#90EE90",
            fg="black",
            height=2
        )
        self.status_label.pack(fill=tk.X)

        # Canvas for game board
        canvas_frame = tk.Frame(main_frame, bg="#F0F0F0")
        canvas_frame.pack(pady=10)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg=self.color_bg,
            highlightthickness=2,
            highlightbackground=self.color_grid
        )
        self.canvas.pack()

        # Draw initial board
        self.draw_grid()
        self.update_board_display()

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.handle_board_click)
        self.canvas.bind("<Motion>", self.handle_mouse_motion)

        # Info panel
        info_frame = tk.Frame(main_frame, bg="#F0F0F0")
        info_frame.pack(fill=tk.X, padx=20, pady=10)

        info_label = tk.Label(
            info_frame,
            text="Game Info:",
            font=("Arial", 12, "bold"),
            bg="#F0F0F0"
        )
        info_label.pack(anchor=tk.W)

        self.info_text = tk.Text(
            info_frame,
            height=4,
            font=("Courier", 10),
            bg="white",
            state=tk.DISABLED
        )
        self.info_text.pack(fill=tk.X)

        # Control buttons
        button_frame = tk.Frame(main_frame, bg="#F0F0F0")
        button_frame.pack(pady=10)

        new_game_btn = tk.Button(
            button_frame,
            text="New Game",
            command=self.show_game_screen,
            font=("Arial", 12),
            padx=15,
            pady=5,
            cursor="hand2"
        )
        new_game_btn.grid(row=0, column=0, padx=5)

        settings_btn = tk.Button(
            button_frame,
            text="Settings",
            command=self.show_settings_screen,
            font=("Arial", 12),
            padx=15,
            pady=5,
            cursor="hand2"
        )
        settings_btn.grid(row=0, column=1, padx=5)

        quit_btn = tk.Button(
            button_frame,
            text="Quit",
            command=self.root.quit,
            font=("Arial", 12),
            bg="#f44336",
            fg="white",
            padx=15,
            pady=5,
            cursor="hand2"
        )
        quit_btn.grid(row=0, column=2, padx=5)

        # Update status and start game
        self.update_status()

        # If bot goes first, trigger bot move
        if not self.is_human_turn:
            self.root.after(500, self.make_bot_move)

    def draw_grid(self):
        """Draw the 9x9 grid with coordinates."""
        # Draw vertical lines
        for i in range(self.board_size + 1):
            x = i * self.cell_size
            self.canvas.create_line(
                x, 0, x, self.canvas_size,
                fill=self.color_grid,
                width=2
            )

        # Draw horizontal lines
        for i in range(self.board_size + 1):
            y = i * self.cell_size
            self.canvas.create_line(
                0, y, self.canvas_size, y,
                fill=self.color_grid,
                width=2
            )

        # Add row and column labels
        for i in range(self.board_size):
            # Column labels (top)
            self.canvas.create_text(
                i * self.cell_size + self.cell_size // 2,
                -10,
                text=str(i),
                font=("Arial", 8),
                fill=self.color_grid
            )
            # Row labels (left)
            self.canvas.create_text(
                -10,
                i * self.cell_size + self.cell_size // 2,
                text=str(i),
                font=("Arial", 8),
                fill=self.color_grid
            )

    def update_board_display(self):
        """Update the visual display of the board."""
        # Remove all pieces (keep grid)
        self.canvas.delete("piece")
        self.canvas.delete("highlight")

        # Draw pieces
        state_2d = self.state.reshape(9, 9)
        for row in range(9):
            for col in range(9):
                if state_2d[row, col] == 1:
                    self.draw_x(row, col)
                elif state_2d[row, col] == -1:
                    self.draw_o(row, col)

    def draw_x(self, row, col):
        """Draw an X at the specified position."""
        x1 = col * self.cell_size + 15
        y1 = row * self.cell_size + 15
        x2 = col * self.cell_size + self.cell_size - 15
        y2 = row * self.cell_size + self.cell_size - 15

        self.canvas.create_line(
            x1, y1, x2, y2,
            fill=self.color_x,
            width=4,
            tags="piece"
        )
        self.canvas.create_line(
            x2, y1, x1, y2,
            fill=self.color_x,
            width=4,
            tags="piece"
        )

    def draw_o(self, row, col):
        """Draw an O at the specified position."""
        cx = col * self.cell_size + self.cell_size // 2
        cy = row * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 2 - 15

        self.canvas.create_oval(
            cx - radius, cy - radius,
            cx + radius, cy + radius,
            outline=self.color_o,
            width=4,
            tags="piece"
        )

    def handle_board_click(self, event):
        """Handle mouse click on the board."""
        # Only process clicks during human's turn
        if not self.is_human_turn or self.game_over:
            return

        # Convert click coordinates to grid position
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        # Validate bounds
        if row < 0 or row >= 9 or col < 0 or col >= 9:
            return

        # Convert to action index
        action_index = row * 9 + col

        # Check if move is valid
        valid_moves = self.game.get_valid_moves(self.state)
        if valid_moves[action_index] != 1:
            # Show invalid move feedback
            self.show_invalid_move_feedback(row, col)
            return

        # Execute move
        self.make_human_move(action_index)

    def handle_mouse_motion(self, event):
        """Handle mouse motion for hover effects."""
        if not self.is_human_turn or self.game_over:
            return

        # Get grid position
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        # Remove previous hover
        self.canvas.delete("hover")

        # Validate bounds
        if row < 0 or row >= 9 or col < 0 or col >= 9:
            return

        # Check if square is empty
        action_index = row * 9 + col
        valid_moves = self.game.get_valid_moves(self.state)

        if valid_moves[action_index] == 1:
            # Draw hover effect
            x1 = col * self.cell_size + 2
            y1 = row * self.cell_size + 2
            x2 = (col + 1) * self.cell_size - 2
            y2 = (row + 1) * self.cell_size - 2

            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=self.color_highlight,
                outline="",
                tags="hover"
            )
            # Lower hover effect behind pieces
            self.canvas.tag_lower("hover")

    def show_invalid_move_feedback(self, row, col):
        """Show visual feedback for invalid move."""
        x1 = col * self.cell_size + 2
        y1 = row * self.cell_size + 2
        x2 = (col + 1) * self.cell_size - 2
        y2 = (row + 1) * self.cell_size - 2

        # Flash red border
        rect = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="red",
            width=3,
            tags="invalid"
        )

        # Remove after 500ms
        self.root.after(500, lambda: self.canvas.delete("invalid"))

    def make_human_move(self, action_index):
        """Process human move."""
        # Update state
        self.state[action_index] = self.current_player

        # Update display
        self.update_board_display()

        # Add to info panel
        row, col = action_index // 9, action_index % 9
        self.add_info(f"You played at ({row}, {col})")

        # Check for game end
        if self.check_game_end():
            return

        # Switch player
        self.current_player *= -1
        self.is_human_turn = False

        # Update status
        self.update_status()

        # Trigger bot move
        self.root.after(300, self.make_bot_move)

    def make_bot_move(self):
        """Initiate bot move in background thread."""
        if self.game_over:
            return

        # Update UI
        self.update_status()
        self.add_info(f"Bot thinking ({self.num_simulations.get()} simulations)...")

        # Disable board interaction
        self.canvas.unbind("<Button-1>")

        # Run MCTS in background thread
        thread = threading.Thread(target=self._bot_move_thread)
        thread.daemon = True
        thread.start()

    def _bot_move_thread(self):
        """Run MCTS in background thread."""
        try:
            # Canonicalize state from bot's perspective
            canonical_state = self.state.copy() * self.current_player

            # Create node for current state
            node = Node(prior_prob=0, player=self.current_player, action_index=None)
            node.set_state(canonical_state)

            # Run MCTS
            root_node = self.mcts.run_simulation(
                root_node=node,
                num_simulations=self.num_simulations.get(),
                player=self.current_player
            )

            # Select best move
            action, _, action_probs = self.mcts.select_move(
                node=root_node,
                mode="exploit",
                temperature=0.1
            )
            action_index = np.argmax(action)

            # Get top moves for display
            visit_counts = np.zeros(cfg.ACTION_SIZE)
            for k, v in root_node.children.items():
                visit_counts[k] = v.total_visits_N

            top_indices = np.argsort(visit_counts)[::-1][:3]

            # Update UI on main thread
            self.root.after(0, lambda: self._apply_bot_move(action_index, top_indices, visit_counts))

        except Exception as e:
            self.root.after(0, lambda: self.handle_bot_error(str(e)))

    def _apply_bot_move(self, action_index, top_indices, visit_counts):
        """Apply bot move to the game state (called on main thread)."""
        # Update state
        self.state[action_index] = self.current_player

        # Update display
        self.update_board_display()

        # Show bot's move and top choices
        row, col = action_index // 9, action_index % 9
        self.add_info(f"Bot played at ({row}, {col})")
        self.add_info("Bot's top 3 moves:")
        for i, idx in enumerate(top_indices, 1):
            if visit_counts[idx] > 0:
                r, c = idx // 9, idx % 9
                self.add_info(f"  {i}. ({r}, {c}): {int(visit_counts[idx])} visits")

        # Check for game end
        if self.check_game_end():
            return

        # Switch player
        self.current_player *= -1
        self.is_human_turn = True

        # Update status
        self.update_status()

        # Re-enable board interaction
        self.canvas.bind("<Button-1>", self.handle_board_click)

    def handle_bot_error(self, error_msg):
        """Handle errors from bot move thread."""
        messagebox.showerror("Bot Error", f"An error occurred during bot move:\n{error_msg}")
        self.show_settings_screen()

    def check_game_end(self):
        """Check if game has ended and handle accordingly."""
        result = self.game.win_or_draw(self.state)

        if result is not None:
            self.game_over = True
            self.canvas.unbind("<Button-1>")
            self.show_game_over_dialog(result)
            return True

        return False

    def show_game_over_dialog(self, result):
        """Show game over dialog."""
        # Create modal dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Game Over")
        dialog.geometry("400x300")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Determine result message and color
        if result == 1:
            if self.human_player.get() == 1:
                message = "Congratulations!\nYou won!"
                color = "#4CAF50"
            else:
                message = "Bot (X) wins!"
                color = "#f44336"
        elif result == -1:
            if self.human_player.get() == -1:
                message = "Congratulations!\nYou won!"
                color = "#4CAF50"
            else:
                message = "Bot (O) wins!"
                color = "#f44336"
        else:
            message = "It's a draw!"
            color = "#9E9E9E"

        # Result label
        result_label = tk.Label(
            dialog,
            text=message,
            font=("Arial", 24, "bold"),
            bg=color,
            fg="white",
            pady=30
        )
        result_label.pack(fill=tk.X)

        # Buttons frame
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

        # Play Again button
        play_again_btn = tk.Button(
            button_frame,
            text="Play Again",
            command=play_again,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        play_again_btn.grid(row=0, column=0, columnspan=2, pady=5, padx=5, sticky=tk.EW)

        # Switch Sides button
        switch_btn = tk.Button(
            button_frame,
            text="Switch Sides",
            command=switch_sides,
            font=("Arial", 12),
            padx=20,
            pady=10,
            cursor="hand2"
        )
        switch_btn.grid(row=1, column=0, pady=5, padx=5, sticky=tk.EW)

        # Main Menu button
        menu_btn = tk.Button(
            button_frame,
            text="Main Menu",
            command=main_menu,
            font=("Arial", 12),
            padx=20,
            pady=10,
            cursor="hand2"
        )
        menu_btn.grid(row=1, column=1, pady=5, padx=5, sticky=tk.EW)

        # Quit button
        quit_btn = tk.Button(
            button_frame,
            text="Quit",
            command=self.root.quit,
            font=("Arial", 12),
            bg="#f44336",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        quit_btn.grid(row=2, column=0, columnspan=2, pady=5, padx=5, sticky=tk.EW)

    def update_status(self):
        """Update the status label."""
        if self.game_over:
            return

        if self.is_human_turn:
            player_symbol = "X" if self.human_player.get() == 1 else "O"
            self.status_label.config(
                text=f"Your turn ({player_symbol})",
                bg="#90EE90"
            )
        else:
            player_symbol = "X" if self.current_player == 1 else "O"
            self.status_label.config(
                text=f"Bot thinking... ({player_symbol})",
                bg="#FFFF99"
            )

    def add_info(self, text):
        """Add text to the info panel."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, text + "\n")
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = TicTacToeGUI()
    app.run()


if __name__ == "__main__":
    main()
