#!/usr/bin/env python3
import argparse
import os

parser = argparse.ArgumentParser(
    prog='Merge',
    description='Merge individual solutions into a single submission file')
parser.add_argument('output_file', type=str, help='Path to output file', default='submission.csv')
args = parser.parse_args()

solution_dir = 'data/solutions'

with open(args.output_file, 'w') as out_file:
    out_file.write('id,moves\n')
    files = sorted(os.listdir(solution_dir), key=lambda x: int(x.split('.')[0]))
    for sol_file_name in files:
        puzzle_id = sol_file_name.split('.')[0]
        with open(f'{solution_dir}/{sol_file_name}', 'r') as sol_file:
            solution_state = sol_file.read()
            out_file.write(f'{puzzle_id},{solution_state}\n')
