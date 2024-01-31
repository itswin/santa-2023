#!/usr/bin/env python3
import argparse
import pandas as pd
import itertools
import numpy as np
import subprocess
from util import *

# seq 160 160 | xargs -P 4 -I {} sh -c "python3 scripts/555_solve.py {}"

def validate_sse_solution(sse_scramble, initial_state, solution_state, partial_sol=None):

    # move_map = get_move_map(n)
    santa_solution = []
    # santa_solution = center_orienting_seq
    for move in sse_scramble:
        # print(move, "\t", sse_to_santa[move])
        santa_solution.append(sse_to_santa[move])

    santa_solution = ".".join(santa_solution).split(".")
    # santa_solution = invert(santa_solution)

    if partial_sol:
        santa_solution = partial_sol + santa_solution

    print(".".join(santa_solution))

    print(f"Validating")
    state = initial_state
    for move_name in santa_solution:
        state = state[moves[move_name]]

    num_difference = evaluate_difference(state, solution_state)
    wildcards = puzzle['num_wildcards']

    print(f"Num difference: {num_difference}")
    print(f"Num wildcards: {wildcards}")

    if num_difference <= wildcards:
        print("Valid solution")
        return True, santa_solution
    else:
        print("Invalid solution")
        return False, santa_solution

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("file", type=str)
parser.add_argument("-ps", "--partial_sol", type=str, default=None)

args = parser.parse_args()

orbit_algs = []
scores = []

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
print(puzzle)

puzzle_type = puzzle["puzzle_type"]
num_wildcards = puzzle["num_wildcards"]
n = int(puzzle_type.split("/")[-1])
moves = get_moves(puzzle["puzzle_type"])

sse_to_santa = get_sse_to_santa_move_map(n)

half_n = n // 2
parity_alg = ""

with open(args.file, "r") as fp:
    orbit = 0
    length = 0
    alg = ""
    for line in fp.readlines():
        if "Parity Algorithm:" in line:
            parity_alg = line.split("Parity Algorithm:")[1].strip()
        elif "Orbit number:" in line:
            orbit = int(line.split("Orbit number:")[1].strip())
        elif "Permutation:" in line:
            perm = line.split("Permutation:")[1].strip()
            perm = perm.replace("(", "").replace(")", "")
            length = len(perm.split())
        elif "Alg:" in line:
            alg = line.split("Alg:")[1].strip()
            # print(alg)
            # print(len(alg.split()))
            # print()
            orbit_algs.append(alg)

            alg_length = len(" ".join(sse_to_santa[move] for move in alg.split()).split())

            # Corners
            if orbit == half_n:
                scores.append(length * 3 / alg_length)
            # Edges
            elif orbit % half_n == 0 and orbit < half_n * half_n:
                scores.append(length * 2 / alg_length)
            else:
                scores.append(length / alg_length)

full_alg = parity_alg + " ".join(orbit_algs)

print(f"Parity alg: {parity_alg}")
print(f"Orbit algs: {orbit_algs}")
print(f"Scores: {scores}")

initial_state = np.array(puzzle["initial_state"].split(";"))
solution_state = np.array(puzzle["solution_state"].split(";"))

if args.partial_sol:
    with open(args.partial_sol, "r") as fp:
        sol = fp.read()
        delimiter = "." if "." in sol else " "
        partial_sol = sol.split(delimiter)

    print(f"Applying partial solution of length {len(partial_sol)}: {partial_sol}")
else:
    partial_sol = None

while True:
    test_alg = parity_alg + " " + " ".join(orbit_algs)
    test_alg = test_alg.split()
    valid, santa_solution = validate_sse_solution(test_alg, initial_state, solution_state, partial_sol)
    if not valid:
        break

    print("Removing an orbit")
    full_alg = test_alg

    worst_index = np.argmax(scores)
    del orbit_algs[worst_index]
    del scores[worst_index]

full_alg = ".".join(full_alg)

with open("data/SSE/console_alg.txt", "w") as fp:
    fp.write(full_alg)

print(f"Length: {len(full_alg.split())}")
print(f"Num orbits: {len(orbit_algs)}")
