#!/usr/bin/env python3
import os

solution_dir = 'data/solutions'

score = 0

files = sorted(os.listdir(solution_dir), key=lambda x: int(x.split('.')[0]))
for sol_file_name in files:
    puzzle_id = sol_file_name.split('.')[0]
    with open(f'{solution_dir}/{sol_file_name}', 'r') as sol_file:
        solution_state = sol_file.read()
        puzzle_score = len(solution_state.split('.'))
        score += puzzle_score
        print(sol_file_name, puzzle_score)

print(f"Total score: {score}")
