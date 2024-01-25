#!/usr/bin/env python3
import argparse
import pandas as pd
import itertools
import numpy as np
from subprocess import Popen, PIPE
import subprocess
from util import *
import threading

# seq 210 234 | xargs -I {} python3 scripts/twsearch_shorten.py {}

def on_timeout(proc):
    print("Timed out")
    proc.kill()

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("--sol_dir", type=str, default="data/solutions")
parser.add_argument("--out_sol_dir", type=str, default="data/solutions")
parser.add_argument("--moves", action="store_true", default=False)
parser.add_argument("--timeout", type=int, default=5 * 60)
parser.add_argument("--decomposed", action="store_false", default=True)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
print(puzzle)

puzzle_type = puzzle["puzzle_type"]
puzzle_type = puzzle_type.replace("/", "_")
n = int(puzzle_type.split("_")[-1])
moves = get_moves(puzzle["puzzle_type"])
print(f"Number of moves: {len(moves)}")

initial_state = puzzle["initial_state"].split(";")
solution_state = puzzle["solution_state"].split(";")

if args.decomposed:
    # Run decompose.py
    subprocess.run(["python3", "scripts/decompose.py", str(args.id), "--unique"])

    tws_file = f"./data/tws_phases/{puzzle_type}/{puzzle_type}_unique_decomposed.tws"
else:
    tws_file = write_tws_file(puzzle, True)

# Use the current solution as a scramble
with open(f"data/solutions/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

is_move_cyclic = get_cyclic_moves(moves)

scramble = " ".join(list(map(create_normalize_inverted_cyclic(is_move_cyclic), current_solution))) + "\n"
print(f"Current solution length: {len(current_solution)}")
print(scramble)

if args.moves:
    # with open("/Users/Win33/Documents/Programming/twsearch/moves.txt", "w") as fp:
    with open("./moves.txt", "w") as fp:
        fp.write(scramble)

SOLVER_PATH = f"/Users/Win33/Documents/Programming/twsearch/build/bin/twsearch -q --microthreads 16 --shortenseqs -M 32768 {tws_file}".split()
p = Popen(SOLVER_PATH, stdout=PIPE, stdin=PIPE)
p.stdin.write(scramble.encode("utf-8"))
p.stdin.flush()

print("Starting timer")
timer = threading.Timer(args.timeout, on_timeout, (p, ))
timer.start()

sol = None
# Search for the solution line
try:
    while True:
        line = p.stdout.readline().decode("utf-8").strip()
        # print(line)
        if not line:
            break
        if line.startswith("FOUND SOLUTION: "):
            sol = line.split(":")[1].strip().split(".")
            print(f"\nNew Solution of length {len(sol)}", sol)
        if "Working with depth 20" in line:
            print("Hit max manual depth")
            p.kill()
            break
except KeyboardInterrupt:
    print("Interrupted")
    p.kill()

print("Last line: ", line)
timer.cancel()

if sol is None:
    print("No solution found")
    exit(1)

if len(sol) < len(current_solution):
    print(f"New solution is shorter than current solution. Writing to file.")
    with open(f"{args.out_sol_dir}/{args.id}.txt", "w") as fp:
        fp.write(".".join(sol))
else:
    print(f"New solution is longer than current solution.")
    print(f"Length of new solution: {len(sol)}")
    print(f"Length of current solution: {len(current_solution)}")
