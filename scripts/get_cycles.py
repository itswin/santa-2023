#!/usr/bin/env python3

import argparse
import pandas as pd
from util import *
import re

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("--partial_sol", type=str, default=None)
parser.add_argument("--use_minus", action="store_true", default=False)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
print(puzzle)

puzzle_type = puzzle["puzzle_type"]
n = int(puzzle_type.split("/")[-1])
moves = get_moves(puzzle["puzzle_type"])
# print(f"Number of moves: {len(moves)}")

initial_state = np.array(puzzle["initial_state"].split(";"))
solution_state = np.array(puzzle["solution_state"].split(";"))

with open(f"data/solutions/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

if args.partial_sol:
    with open(args.partial_sol, "r") as fp:
        sol = fp.read()
        delimiter = "." if "." in sol else " "
        partial_sol = sol.split(delimiter)

    print(f"Applying partial solution of length {len(partial_sol)}: {partial_sol}")
else:
    partial_sol = []

state = initial_state.copy()
for move in partial_sol:
    state = state[moves[move]]

print("Initial state: ", initial_state)

# Only use on unique sol states
cycles = []
solved = set()
for index, piece in enumerate(state):
    cycle = []
    desired_index = int(piece[1:])
    if desired_index in solved:
        continue
    while index != desired_index and desired_index not in cycle:
        cycle.append(index)
        solved.add(index)
        index = desired_index
        desired_index = int(state[index][1:])
    cycle.append(index)
    cycles.append(cycle)

for cycle in cycles:
    print(cycle)
