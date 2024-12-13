# Tic Tac Toe AI

This project is an implementation of the game "Tic-Tac-toe" with artificial intelligence. You can train AI models and play saved games.

## Project Launch

The project is launched via the command line using the file `main.py `. 

### Basic parameters:

- `-m`, `--mode`: Operating mode (train or replay)
- `-g`, `--generations`: Number of generations to train
- `-p`, `--population`: The size of the population in each generation
- `-r`, `--rounds-gen`: Number of rounds per generation
- `-b`, `--board-size`: The size of the playing field
- `-w`, `--win-line`: The length of the line to win
- `-v`, `--visualize`: Visualization of the playing field
- `-d`, `--move-delay`: Delay between moves in seconds
- `-n`, `--number`: The number of the game to play

## Usage examples

### 1. AI training with default settings:

```
python main.py
```

This will start a training session with 100 generations, a population of 10, 5 rounds per generation, on a 3x3 board with a 3-length victory line.

### 2. AI training with custom settings:

```
python main.py -m train -g 200 -p 20 -r 10 -b 4 -w 4 -v -d 0.5
```

This command:
- Will start a training session (`-m train`)
- With 200 generations (`-g 200`)
- A population of 20 in each generation (`-p 20`)
- 10 rounds per generation (`-r 10`)
- On a 4x4 board (`-b 4`)
- With a 4-length victory line (`-w 4`)
- With a visualization of the playing field (`-v`)
- With a delay between moves of 0.5 seconds (`-d 0.5`)

### 3. Playing a saved game:

```
python main.py -m replay -n 5 -b 3 -w 3 -d 1
```

This command:
- Starts the replay mode (`-m replay`)
- Plays game number 5 (`-n 5`)
- On a 3x3 board (`-b 3`)
- With a 3-length win line (`-w 3`)
- With a delay of 1 second between moves (`-d 1`)

### 4. Training on a large board:

```
python main.py -m train -g 500 -p 30 -r 20 -b 5 -w 4 -v
```

This command will start a long training session on a 5x5 board with a 4-length victory line.

### 5. Fast training without visualization:

```
python main.py -m train -g 50 -p 5 -r 3 -d 0.0001
```

This command will do a quick workout without visualization and with minimal delay between moves.

### 6. Check the statistics:

```
python plot_training_results.py
```

### 7. Playing with your AI:

```
python play_game.py
```

## Tips for using

1. For a quick workout, reduce the number of generations, population size, and number of rounds.
2. Increasing the size of the board and the length of the line to win significantly complicates the game and may require more time to practice.
3. Visualization (`-v') is useful for observing the process, but slows down the workout. Use it mainly when playing games or with a small number of generations.
4. Experiment with different parameters to find the optimal balance between the speed of training and the quality of AI training.
