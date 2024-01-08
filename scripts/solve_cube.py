#!/usr/bin/env python3
import argparse
import pandas as pd
import itertools
import numpy as np
import subprocess
from util import *

# seq 210 234 | xargs -P 4 -I {} python3 scripts/solve_odd.py {}

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("--sol_dir", type=str, default="data/solutions")
parser.add_argument("--out_sol_dir", type=str, default="data/solutions")
parser.add_argument("--partial_sol", type=str, default=None)
parser.add_argument("--show_faces_only", action="store_true", default=False)

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

if args.partial_sol:
    with open(args.partial_sol, "r") as fp:
        center_orienting_seq = fp.read().split(".")
    for move_name in center_orienting_seq:
        state = state[moves[move_name]]
    print("PARTIAL", state)
else:
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

if args.show_faces_only:
    exit()

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

mapped_sol = center_orienting_seq + mapped_sol
formatted_sol = ".".join(mapped_sol)
mapped_sol = formatted_sol.split(".")
print(formatted_sol)

current_solution = []
with open(f"{args.sol_dir}/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

if len(mapped_sol) < len(current_solution):
    print(f"New solution is shorter than current solution. Writing to file.")
    print(f"Length of new solution: {len(mapped_sol)}")
    print(f"Length of current solution: {len(current_solution)}")
    with open(f"{args.out_sol_dir}/{args.id}.txt", "w") as fp:
        fp.write(formatted_sol)
else:
    print(f"New solution is longer than current solution.")
    print(f"Length of new solution: {len(mapped_sol)}")
    print(f"Length of current solution: {len(current_solution)}")
