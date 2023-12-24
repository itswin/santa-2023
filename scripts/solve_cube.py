#!/usr/bin/env python3
import argparse
import pandas as pd
from get_moves import get_moves
import itertools
import numpy as np
import subprocess
from util import *

# seq 210 234 | xargs -P 4 -I {} python3 scripts/solve_odd.py {}

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
print(puzzle)

puzzle_type = puzzle["puzzle_type"]
n = int(puzzle_type.split("/")[-1])
moves = get_moves(puzzle["puzzle_type"])
print(f"Number of moves: {len(moves)}")

initial_state = puzzle["initial_state"].split(";")
solution_state = puzzle["solution_state"].split(";")

STICKER_MAP = {
    'A': 'U',
    'B': 'F',
    'C': 'R',
    'D': 'B',
    'E': 'L',
    'F': 'D',
}

state = np.array(initial_state)
print(state)

print("INITIAL", state)

if n % 2 == 0:
    center_orienting_seq = []
else:
    state, center_orienting_seq = orient_centers(state, moves, n)

print("ORIENTED", state)
print("ORIENT_CENTERS", center_orienting_seq)

state = "".join(STICKER_MAP[c] for c in state)
faces = state_to_faces(state, n)

print_faces(faces, n)

cubestring = make_cubestring(faces)
print(cubestring)

move_map = get_move_map(n)
print(move_map)

SOLVER_PATH = "/Users/Win33/Documents/Programming/rubiks-cube-NxNxN-solver/rubiks-cube-solver.py"
cmd = [SOLVER_PATH, "--state", cubestring]
# cmd = ["cat", "scripts/split.py"]

out = subprocess.check_output(cmd)

out = out.decode("utf-8").strip()
out = out.split("\n")

# Search for the solution line
for line in out:
    if line.startswith("Solution: "):
        sol = line.split(":")[1].strip()
        break

print(sol)

# Map it back to our move set
mapped_sol = []
for move in sol.split(" "):
    mapped_sol.append(move_map[move])

mapped_sol = (".".join(center_orienting_seq) + "." if len(center_orienting_seq) > 0 else "") + ".".join(mapped_sol)
print(mapped_sol)

# Write it to the solution file
with open(f"data/solutions/{args.id}.txt", "w") as f:
    f.write(mapped_sol)
