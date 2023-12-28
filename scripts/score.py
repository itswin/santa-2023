#!/usr/bin/env python3
import math
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--sol_dir", type=str, default='data/solutions')
args = parser.parse_args()

solution_dir = args.sol_dir

score = 0

puzzle_total = {}
puzzles = pd.read_csv("data/puzzles.csv").set_index("id")

puzzle_ids = []
puzzle_sol_lens = []

files = sorted(os.listdir(solution_dir), key=lambda x: int(x.split('.')[0]))
for sol_file_name in files:
    puzzle_id = sol_file_name.split('.')[0]
    with open(f'{solution_dir}/{sol_file_name}', 'r') as sol_file:
        solution_state = sol_file.read()
        puzzle_score = len(solution_state.split('.'))
        score += puzzle_score
        puzzle = puzzles.loc[int(puzzle_id)]
        puzzle_type = puzzle["puzzle_type"]
        print(sol_file_name, puzzle_score, puzzle_type)
        puzzle_ids.append(puzzle_id)
        puzzle_sol_lens.append(math.log(puzzle_score))
        if puzzle_type not in puzzle_total:
            puzzle_total[puzzle_type] = 0
        puzzle_total[puzzle_type] += puzzle_score

# By puzzle type
print("\nBy puzzle type")
for puzzle_type in puzzle_total:
    print(f"{puzzle_type}: {puzzle_total[puzzle_type]}")

print(f"\nTotal score: {score}")

# Plot the scores
plt.figure(figsize=(10, 10))
plt.plot(puzzle_ids, puzzle_sol_lens)
plt.xlabel('Puzzle ID')
plt.ylabel('Solution Length')
plt.title('Solution Lengths')
plt.savefig('solution_lengths.png')
