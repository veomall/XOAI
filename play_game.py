import sys
import os
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QMessageBox, QComboBox

from ai.neural_net import TicTacToeNet


def get_available_models():
    models_dir = 'data/saved_models'
    models = []
    for filename in os.listdir(models_dir):
        if filename.startswith('model_') and filename.endswith('_to_win.pth'):
            parts = filename[6:-11].split('x')
            if len(parts) == 2:
                # Проверяем, начинается ли первая часть с 'X_' или 'O_'
                if parts[0].startswith(('X_', 'O_')):
                    board_size = parts[0][2:]  # Убираем 'X_' или 'O_'
                else:
                    board_size = parts[0]
                win_line = parts[1].split('_')[0]
                try:
                    models.append((int(board_size), int(win_line)))
                except ValueError:
                    # Пропускаем файлы с некорректным форматом
                    continue
    return sorted(set(models))


class GameBoard:
    def __init__(self, size, win_line):
        self.size = size
        self.win_line = win_line
        self.board = np.zeros((size, size), dtype=int)

    def make_move(self, x, y, player):
        if self.board[x, y] == 0:
            self.board[x, y] = player
            return True, self.check_winner()
        return False, None

    def check_winner(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] != 0:
                    if self.check_line(i, j, 1, 0) or \
                       self.check_line(i, j, 0, 1) or \
                       self.check_line(i, j, 1, 1) or \
                       self.check_line(i, j, 1, -1):
                        return self.board[i, j]
        return None

    def check_line(self, x, y, dx, dy):
        player = self.board[x, y]
        for i in range(self.win_line):
            if x < 0 or x >= self.size or y < 0 or y >= self.size or self.board[x, y] != player:
                return False
            x += dx
            y += dy
        return True

    def is_full(self):
        return np.all(self.board != 0)

    def get_valid_moves(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]


class GameAIAgent:
    def __init__(self, model):
        self.model = model

    def get_action(self, board):
        state = torch.FloatTensor(board.board).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        valid_moves = board.get_valid_moves()
        valid_q_values = [q_values[0][i * board.size + j] for i, j in valid_moves]
        return valid_moves[np.argmax(valid_q_values)]


class TicTacToeGame(QWidget):
    def __init__(self):
        super().__init__()
        self.board = None
        self.model_x = None
        self.model_o = None
        self.ai_agent = None
        self.human_first = True
        self.buttons = []
        self.available_models = get_available_models()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Tic Tac Toe vs AI')
        layout = QVBoxLayout()

        # Add dropdown for game mode selection
        self.mode_combo = QComboBox()
        for size, win_line in self.available_models:
            self.mode_combo.addItem(f"{size}x{size} (Win: {win_line})")
        layout.addWidget(QLabel('Select game mode:'))
        layout.addWidget(self.mode_combo)

        # Add button to start game
        start_button = QPushButton('Start Game')
        start_button.clicked.connect(self.setup_game)
        layout.addWidget(start_button)

        # Add a label to show game status
        self.status_label = QLabel('Choose game mode and start the game')
        layout.addWidget(self.status_label)

        self.game_layout = QVBoxLayout()
        layout.addLayout(self.game_layout)

        self.setLayout(layout)
        self.show()

    def setup_game(self):
        # Clear previous game if exists
        for i in reversed(range(self.game_layout.count())): 
            self.game_layout.itemAt(i).widget().setParent(None)

        selected_mode = self.mode_combo.currentText()
        board_size = int(selected_mode.split('x')[0])
        win_line = int(selected_mode.split('(Win: ')[1].split(')')[0])

        self.board = GameBoard(board_size, win_line)
        model_x_path = f'data/saved_models/model_X_{board_size}x{board_size}_{win_line}_to_win.pth'
        model_o_path = f'data/saved_models/model_O_{board_size}x{board_size}_{win_line}_to_win.pth'

        try:
            self.model_x = TicTacToeNet(board_size)
            self.model_x.load_state_dict(torch.load(model_x_path))
            self.model_x.eval()

            self.model_o = TicTacToeNet(board_size)
            self.model_o.load_state_dict(torch.load(model_o_path))
            self.model_o.eval()
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"Model file not found: {model_x_path} or {model_o_path}")
            return

        # Create grid for the game board
        grid = QGridLayout()
        self.buttons = []
        for i in range(board_size):
            row = []
            for j in range(board_size):
                button = QPushButton('')
                button.setFixedSize(50, 50)
                button.clicked.connect(lambda _, x=i, y=j: self.on_click(x, y))
                grid.addWidget(button, i, j)
                row.append(button)
            self.buttons.append(row)

        self.game_layout.addLayout(grid)

        # Add buttons for choosing turn
        turn_layout = QHBoxLayout()
        first_button = QPushButton('Play First')
        second_button = QPushButton('Play Second')
        first_button.clicked.connect(lambda: self.start_game(True))
        second_button.clicked.connect(lambda: self.start_game(False))
        turn_layout.addWidget(first_button)
        turn_layout.addWidget(second_button)
        self.game_layout.addLayout(turn_layout)

        self.status_label.setText('Choose your turn:')

    def start_game(self, human_first):
        self.human_first = human_first
        self.board = GameBoard(self.board.size, self.board.win_line)
        for row in self.buttons:
            for button in row:
                button.setText('')
        self.update_board()
        self.status_label.setText("Your turn" if human_first else "AI's turn")

        # Выбор соответствующей модели для AI
        if human_first:
            self.ai_agent = GameAIAgent(self.model_o)
        else:
            self.ai_agent = GameAIAgent(self.model_x)
        if not human_first:
            self.ai_move()



    def on_click(self, x, y):
        if self.board is None:
            return
        if self.board.board[x, y] == 0:
            player = 1 if self.human_first else 2
            valid, winner = self.board.make_move(x, y, player)
            if valid:
                self.update_board()
                if winner:
                    self.end_game(winner)
                elif self.board.is_full():
                    self.end_game(0)
                else:
                    self.status_label.setText("AI's turn")
                    self.ai_move()

    def ai_move(self):
        valid_moves = self.board.get_valid_moves()
        if not valid_moves:
            self.end_game(0)
            return
        x, y = self.ai_agent.get_action(self.board)
        player = 2 if self.human_first else 1
        valid, winner = self.board.make_move(x, y, player)
        if valid:
            self.update_board()
            if winner:
                self.end_game(winner)
            elif self.board.is_full():
                self.end_game(0)
            else:
                self.status_label.setText("Your turn")

    def update_board(self):
        for i in range(self.board.size):
            for j in range(self.board.size):
                if self.board.board[i, j] == 1:
                    self.buttons[i][j].setText('X')
                elif self.board.board[i, j] == 2:
                    self.buttons[i][j].setText('O')
                else:
                    self.buttons[i][j].setText('')


    def end_game(self, winner):
        if winner == 0:
            QMessageBox.information(self, "Game Over", "It's a draw!")
        else:
            winner_text = "You win!" if (winner == 1) == self.human_first else "AI wins!"
            QMessageBox.information(self, "Game Over", winner_text)
        self.status_label.setText('Choose your turn:')
        for row in self.buttons:
            for button in row:
                button.setText('')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TicTacToeGame()
    sys.exit(app.exec_())
