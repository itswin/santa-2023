#! /usr/bin/env python3
import re
import argparse

parser = argparse.ArgumentParser(description='Converts a file from one format to another.')
parser.add_argument('input', type=str, help='Input file')

args = parser.parse_args()

with open(args.input, 'r') as f:
    lines = f.readlines()
    moves_found = []
    moves = set(lines[0].strip().split(' '))

    print(f"Number of moves: {len(moves)}")
    print(f"Moves: {moves}")
