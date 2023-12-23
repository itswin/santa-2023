#!/usr/bin/env python3
import argparse
import pandas as pd
import os

data_folder = "data"
subfolders = [ f.path for f in os.scandir(data_folder) if f.is_dir() ]

# Remove "data/solutions" from subfolders and place it at the beginning
subfolders.remove(f'{data_folder}/solutions')
subfolders.insert(0, f'{data_folder}/solutions')

# Find the best solution for each puzzle
best_solution = {}
for folder in subfolders:
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            puzzle_id = file.split('.')[0]
            with open(f'{folder}/{file}', 'r') as sol_file:
                solution_state = sol_file.read()
                puzzle_score = len(solution_state.split('.'))
                if puzzle_id not in best_solution:
                    best_solution[puzzle_id] = (puzzle_score, f'{folder}/{file}')
                else:
                    if puzzle_score < best_solution[puzzle_id][0]:
                        best_solution[puzzle_id] = (puzzle_score, f'{folder}/{file}')

# Overwrite the best solution for each puzzle in data/solutions
for puzzle_id, (score, file) in best_solution.items():
    if file.startswith(f'{data_folder}/solutions'):
        continue
    with open(f'{data_folder}/solutions/{puzzle_id}.txt', 'w') as sol_file:
        with open(file, 'r') as best_sol_file:
            solution_state = best_sol_file.read()
            sol_file.write(solution_state)
