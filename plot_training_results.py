import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def find_csv_files():
    csv_files = glob.glob('data/training_logs/*.csv')
    return csv_files

def select_csv_file(csv_files):
    print("Available CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    while True:
        try:
            choice = int(input("Enter the number of the file you want to plot: "))
            if 1 <= choice <= len(csv_files):
                return csv_files[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
def plot_training_results(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Generation'], df['X Wins'], label='X Wins', color='red')
    plt.plot(df['Generation'], df['O Wins'], label='O Wins', color='blue')
    plt.plot(df['Generation'], df['Draws'], label='Draws', color='green')

    # Customize the plot
    plt.title(f'Training Results - {os.path.basename(csv_file)}')
    plt.xlabel('Generation')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    csv_files = find_csv_files()

    if not csv_files:
        print("No CSV files found in the data/training_logs directory.")
        exit(1)

    selected_file = select_csv_file(csv_files)
    plot_training_results(selected_file)