import argparse
from game.game import Game
from ai.trainer import Trainer
from replay import Replay

def main():
    parser = argparse.ArgumentParser(description="Tic Tac Toe AI - Train AI models or replay saved games")
    parser.add_argument("-m", "--mode", choices=["train", "replay"], default="train", help="Mode: train AI or replay a game (default: train)")

    parser.add_argument("-g", "--generations", type=int, default=100, help="Number of training generations (default: 100)")
    parser.add_argument("-p", "--population", type=int, default=10, help="Population of generation (default: 10)")
    parser.add_argument("-r", "--rounds-gen", type=int, default=5, help="Rounds in one generation (default: 5)")
    parser.add_argument("-b", "--board-size", type=int, default=3, help="Size of the board (default: 3)")
    parser.add_argument("-w", "--win-line", type=int, default=3, help="Win line's length (default: 3)")
    
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize the game board")
    parser.add_argument("-d", "--move-delay", type=float, default=0.001, help="Delay between moves in seconds (default: 0.001)")

    parser.add_argument("-n", "--number", type=int, help="Game number to replay")
    args = parser.parse_args()

    if args.mode == "train":
        trainer = Trainer(delay=args.move_delay, board_size=args.board_size, win_line=args.win_line, population=args.population, visualize=args.visualize)
        trainer.train(args.generations, args.rounds_gen)
    elif args.mode == "replay":
        if not args.number:
            print("Please specify a game number to replay with -n or --number")
            return
        replay = Replay(args.number, delay=args.move_delay, board_size=args.board_size, win_line=args.win_line)
        replay.start()

if __name__ == "__main__":
    main()
