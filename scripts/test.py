#!/usr/bin/env python3
import argparse
import pandas as pd
import os
import numpy as np
from util import *

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("moves", type=str)

args = parser.parse_args()

with open(args.moves, "r") as fp:
    new_moves = fp.read().split()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]

puzzle_type = puzzle["puzzle_type"]
n = int(puzzle_type.split("/")[-1])
moves = get_moves(puzzle["puzzle_type"])

initial_state = puzzle["initial_state"].split(";")
solution_state = puzzle["solution_state"].split(";")

with open(f"data/solutions/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

move_map = get_move_map(n)

state = np.array(initial_state)

santa_moves = [move_map[move] for move in new_moves]
santa_moves = ".".join(santa_moves).split(".")

for move_name in santa_moves:
    state = state[moves[move_name]]

difference = np.count_nonzero(solution_state != state)

print(f"Number of different stickers: {difference}")
