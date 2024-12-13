import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from game.visualizer import Visualizer
from game.board import Board

class Replay:
    def __init__(self, game, delay, board_size, win_line):
        self.moves = np.load(f"data/games/game_{game}.npy")
        self.visualizer = Visualizer(board_size, delay)
        self.board = Board(board_size, win_line)
        self.current_move = 0
        self.result = self.moves[0][0]  # Get the game result from the first row
        self.board_size = board_size

    def start(self):
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(5, 5))  # Increased figure height
        plt.subplots_adjust(bottom=0.2, top=0.9)  # Adjusted top margin

        self.prev_button_ax = plt.axes([0.2, 0.05, 0.25, 0.075])
        self.next_button_ax = plt.axes([0.55, 0.05, 0.25, 0.075])
        self.prev_button = Button(self.prev_button_ax, 'Previous')
        self.next_button = Button(self.next_button_ax, 'Next')

        self.prev_button.on_clicked(self.prev_move)
        self.next_button.on_clicked(self.next_move)

        # Add text for game result
        self.result_text = self.fig.text(0.5, 0.97, self.get_result_text(), 
                                         ha='center', va='center', fontsize=12, fontweight='bold')

        self.update_board()

        while plt.fignum_exists(self.fig.number):
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(0.05)

    def update_board(self):
        self.ax.clear()
        move = self.moves[self.current_move + 1]  # +1 to skip the result row
        self.board.board = np.zeros((self.board_size, self.board_size), dtype=int)
        for m in self.moves[1:self.current_move + 2]:  # Apply all moves up to current
            self.board.board[m[0], m[1]] = m[2]
        result = self.board.check_winner()
        winner, winning_line = result if result else (None, None)
        self.visualizer.draw_board(self.board, winner, winning_line, ax=self.ax)
        self.ax.set_title(f"Move: {self.current_move + 1}/{len(self.moves) - 1}")
        self.fig.canvas.draw()

    def prev_move(self, event):
        if self.current_move > 0:
            self.current_move -= 1
            self.update_board()

    def next_move(self, event):
        if self.current_move < len(self.moves) - 2:  # -2 because of the result row
            self.current_move += 1
            self.update_board()

    def get_result_text(self):
        if self.result == 1:
            return "X wins!"
        elif self.result == 2:
            return "O wins!"
        else:
            return "It's a draw!"

