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
parser.add_argument("--partial_sol", type=str)
parser.add_argument("--sol_dir", type=str, default="data/solutions")
parser.add_argument("--out_sol_dir", type=str, default="data/solutions")
parser.add_argument("--moves", action="store_true", default=False)
parser.add_argument("--unique", action="store_true", default=False)
parser.add_argument("--commutator_file", type=str, default=None)
parser.add_argument("--tws_file", type=str, default=None)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
print(puzzle)

puzzle_type = puzzle["puzzle_type"]
n = int(puzzle_type.split("/")[-1])
moves = get_moves(puzzle["puzzle_type"])
print(f"Number of moves: {len(moves)}")

initial_state = puzzle["initial_state"].split(";")
solution_state = puzzle["solution_state"].split(";")

commutators = None
if args.commutator_file:
    commutators = create_commutators(args.commutator_file, moves)

if args.tws_file:
    tws_file = args.tws_file
else:
    tws_file = write_tws_file(puzzle, args.unique, commutators)

# Use the current solution as a scramble
with open(f"data/solutions/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

is_move_cyclic = get_cyclic_moves(moves)

# Invert the current solution as a scramble
scramble = " ".join(reversed(list(map(create_invert_if_not_cycle(is_move_cyclic), current_solution))))

if args.partial_sol:
    with open(args.partial_sol, "r") as fp:
        partial_solution = fp.read().strip()

    if "." in partial_solution:
        partial_solution = partial_solution.split(".")
    else:
        partial_solution = partial_solution.split(" ")

    partial_solution = list(map(create_normalize_inverted_cyclic(is_move_cyclic), partial_solution))
    scramble = scramble + " " + " ".join(partial_solution)

    print(f"Applying partial sol of length {len(partial_solution)}")
else:
    partial_solution = None

print(scramble)

if args.moves:
    # with open("/Users/Win33/Documents/Programming/twsearch/moves.txt", "w") as fp:
    with open("./moves.txt", "w") as fp:
        fp.write(scramble)
    exit()

SOLVER_PATH = f"/Users/Win33/Documents/Programming/twsearch/build/bin/twsearch -q -s -M 32768 {tws_file}".split()
p = Popen(SOLVER_PATH, stdout=PIPE, stdin=PIPE, stderr=PIPE)
out = p.communicate(input=scramble.encode())[0]

p.wait()

out = out.decode("utf-8").strip()
out = out.split("\n")

sol = None
# Search for the solution line
for line in out:
    if line.startswith("FOUND SOLUTION: "):
        sol = line.split(":")[1].strip()
        break

if not sol:
    print("No solution found.")
    print(out)

    # Print stderr
    print(p.stderr.read().decode("utf-8"))
    exit()

sol = sol.split(".")

if partial_solution:
    sol = partial_solution + sol

if len(sol) < len(current_solution):
    print(f"New solution is shorter than current solution. Writing to file.")
    print(f"Length of new solution: {len(sol)}")
    print(f"Length of current solution: {len(current_solution)}")
    with open(f"{args.out_sol_dir}/{args.id}.txt", "w") as fp:
        fp.write(".".join(sol))
else:
    print(f"New solution is longer than current solution.")
    print(f"Length of new solution: {len(sol)}")
    print(f"Length of current solution: {len(current_solution)}")
