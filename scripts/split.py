#!/usr/bin/env python3
import argparse
import os

parser = argparse.ArgumentParser(
    prog='Split',
    description='Split submission file into individual solutions')
parser.add_argument('submission_file', type=str, help='Path to submission file')
parser.add_argument('output_dir', type=str, help='Path to output directory')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

with open(args.submission_file, 'r') as f:
    for line in f:
        puzzle_id, solution_state = line.strip().split(',')
        if puzzle_id == 'id':
            continue
        with open(f'{args.output_dir}/{puzzle_id}.txt', 'w+') as f:
            f.write(solution_state)
