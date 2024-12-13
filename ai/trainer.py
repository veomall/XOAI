import torch
import torch.optim as optim
import torch.nn.functional as F
import csv
import os
import random
import numpy as np
from collections import deque
from .neural_net import TicTacToeNet
from .agents import AIAgent
from game.game import Game

class Trainer:
    def __init__(self, delay, board_size, win_line, population, visualize=False):
        self.delay = delay
        self.board_size = board_size
        self.win_line = win_line
        self.population_size = population
        self.visualize = visualize

        self.mutation_rate = 0.1
        self.weak_mutation_strength = 0.01
        self.strong_mutation_strength = 0.1
        self.strong_mutation_chance = 0.05

        self.x_models = [TicTacToeNet(self.board_size) for _ in range(self.population_size)]
        self.o_models = [TicTacToeNet(self.board_size) for _ in range(self.population_size)]

        self.x_agents = [AIAgent(model, epsilon=0.05) for model in self.x_models]
        self.o_agents = [AIAgent(model, epsilon=0.05) for model in self.o_models]

    def set_models_to_eval(self):
        for model in self.x_models + self.o_models:
            model.eval()

    def train(self, generations, games_per_generation):
        self.set_models_to_eval()
        os.makedirs('data/training_logs', exist_ok=True)
        csv_file = f'data/training_logs/training_log_{self.board_size}x{self.board_size}_{self.win_line}_to_win.csv'

        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Generation', 'X Wins', 'O Wins', 'Draws'])

            for generation in range(1, generations + 1):
                x_scores = [0] * self.population_size
                o_scores = [0] * self.population_size
                x_wins, o_wins, draws = 0, 0, 0

                for _ in range(games_per_generation):
                    for i in range(self.population_size):
                        for j in range(self.population_size):
                            game = Game(self.delay, self.board_size, self.win_line, visualize=self.visualize)
                            winner, moves = game.play(self.x_agents[i], self.o_agents[j])

                            max_moves = self.board_size * self.board_size
                            move_score = max(1, (max_moves - moves) / max_moves * 10)  # Оценка от 1 до 10

                            if winner == 1:  # X wins
                                x_scores[i] += move_score * 2  # Бонус за победу
                                o_scores[j] -= 5  # Штраф за проигрыш
                                x_wins += 1
                            elif winner == 2:  # O wins
                                o_scores[j] += move_score * 3  # Больший бонус за победу, так как это сложнее
                                x_scores[i] -= 5  # Штраф за проигрыш
                                o_wins += 1
                            else:  # Draw
                                x_scores[i] += 1  # Небольшой бонус за ничью
                                o_scores[j] += 5  # Значительный бонус за ничью
                                draws += 1

                # Evolution step
                self.evolve_population(self.x_models, x_scores)
                self.evolve_population(self.o_models, o_scores, is_o=True)

                # Update agents with new models
                self.x_agents = [AIAgent(model, epsilon=0.05) for model in self.x_models]
                self.o_agents = [AIAgent(model, epsilon=0.05) for model in self.o_models]

                print(f"Generation {generation}: X wins: {x_wins}, O wins: {o_wins}, draws: {draws}")
                writer.writerow([generation, x_wins, o_wins, draws])

        # Save the best models
        os.makedirs('data/saved_models', exist_ok=True)
        torch.save(self.x_models[0].state_dict(), f'data/saved_models/model_X_{self.board_size}x{self.board_size}_{self.win_line}_to_win.pth')
        torch.save(self.o_models[0].state_dict(), f'data/saved_models/model_O_{self.board_size}x{self.board_size}_{self.win_line}_to_win.pth')


    def evolve_population(self, models, scores, is_o=False):
        # Sort models by their scores in descending order
        sorted_models_scores = sorted(zip(models, scores), key=lambda x: x[1], reverse=True)

        # Calculate min and max scores
        min_score = sorted_models_scores[-1][1]
        max_score = sorted_models_scores[0][1]

        # Calculate the adaptive threshold
        if is_o:
            threshold_percentage = 0.8  # Более жесткий отбор для O
        else:
            threshold_percentage = 0.7  # Для X оставляем как было

        score_threshold = min_score + (max_score - min_score) * threshold_percentage
        # Keep all models above the threshold
        top_performers = [model for model, score in sorted_models_scores if score >= score_threshold]

        # If we have less than 2 top performers, keep at least the top 2
        if len(top_performers) < 2:
            top_performers = [model for model, _ in sorted_models_scores[:2]]

        new_models = []

        # Keep the top performers
        new_models.extend(top_performers)

        # Fill the rest of the population with offspring of top performers
        while len(new_models) < self.population_size:
            parent1, parent2 = random.sample(top_performers, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_models.append(child)

        # Update the population
        models[:] = new_models

        # Print some information about the selection process
        print(f"{'O' if is_o else 'X'} Selection: Min score: {min_score:.2f}, Max score: {max_score:.2f}, Threshold: {score_threshold:.2f}")
        print(f"Selected {len(top_performers)} out of {self.population_size} models")


    def crossover(self, parent1, parent2):
        child = TicTacToeNet(self.board_size)
        child_state_dict = child.state_dict()
        p1_state_dict = parent1.state_dict()
        p2_state_dict = parent2.state_dict()

        for key in child_state_dict:
            if random.random() < 0.5:
                child_state_dict[key] = p1_state_dict[key]
            else:
                child_state_dict[key] = p2_state_dict[key]

        child.load_state_dict(child_state_dict)
        return child

    def mutate(self, model):
        with torch.no_grad():
            for param in model.parameters():
                if random.random() < self.mutation_rate:
                    if random.random() < self.strong_mutation_chance:
                        param += torch.randn(param.size()) * self.strong_mutation_strength
                    else:
                        param += torch.randn(param.size()) * self.weak_mutation_strength
