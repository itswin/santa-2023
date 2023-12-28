#!/usr/bin/env python3
import argparse
import pandas as pd
import itertools
import numpy as np
from subprocess import Popen, PIPE
import subprocess
from util import *

# seq 210 234 | xargs -P 4 -I {} python3 scripts/solve_twsearch.py {}

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("--sol_dir", type=str, default="data/solutions")
parser.add_argument("--out_sol_dir", type=str, default="data/solutions")

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
print(puzzle)

puzzle_type = puzzle["puzzle_type"]
n = int(puzzle_type.split("/")[-1])
moves = get_moves(puzzle["puzzle_type"])
print(f"Number of moves: {len(moves)}")

initial_state = puzzle["initial_state"].split(";")
solution_state = puzzle["solution_state"].split(";")

tws_file = write_tws_file(puzzle)

# Use the current solution as a scramble
with open(f"data/solutions/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

scramble = " ".join(reversed(list(map(invert, current_solution))))
print(scramble)

SOLVER_PATH = f"/Users/Win33/Documents/Programming/twsearch/build/bin/twsearch -q -s -M 32768 {tws_file}".split()
p = Popen(SOLVER_PATH, stdout=PIPE, stdin=PIPE, stderr=PIPE)
out = p.communicate(input=scramble.encode())[0]

p.wait()

out = out.decode("utf-8").strip()
out = out.split("\n")

# Search for the solution line
for line in out:
    if line.startswith("FOUND SOLUTION: "):
        sol = line.split(":")[1].strip()
        break

sol = sol.split(".")

if len(sol) < len(current_solution):
    print(f"New solution is shorter than current solution. Writing to file.")
    with open(f"{args.out_sol_dir}/{args.id}.txt", "w") as fp:
        fp.write(".".join(sol))
else:
    print(f"New solution is longer than current solution.")
    print(f"Length of new solution: {len(sol)}")
    print(f"Length of current solution: {len(current_solution)}")
