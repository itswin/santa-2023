#!/usr/bin/env python3
import argparse
import pandas as pd
from get_moves import get_moves

parser = argparse.ArgumentParser()
parser.add_argument("--problem_id", type=int, required=True)
parser.add_argument("--sol", type=str, required=True)
args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.problem_id]

with open(args.sol, "r") as fp:
    current_solution = fp.read().split(".")

print(puzzle)
print(f"Sample score: {len(current_solution)}")

moves = get_moves(puzzle["puzzle_type"])
print(f"Number of moves: {len(moves)}")

state = puzzle["initial_state"].split(";")
for move_name in current_solution:
    state = [state[i] for i in moves[move_name]]
print(state)
assert puzzle["solution_state"].split(";") == state

print("Solution is correct")
