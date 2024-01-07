#!/usr/bin/env python3
import argparse
import pandas as pd
import os
import numpy as np
from util import *

def validate_one(id, sol_file_name, verbose=False):
    puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[id]

    with open(sol_file_name, 'r') as sol_file:
        solution_state = sol_file.read()
        solution = solution_state.split('.')
        if verbose:
            print(f"Solution score: {len(solution)}")
            print(f"Solution: {solution}")

    moves = get_moves(puzzle["puzzle_type"])
    if verbose:
        print(f"Number of moves: {len(moves)}")

    num_wildcards = puzzle["num_wildcards"]

    state = np.array(puzzle["initial_state"].split(";"))
    if verbose:
        print(f"Initial state: {state}")
        print(f"Solution state: {puzzle['solution_state']}")

    for move_name in solution:
        state = state[moves[move_name]]

    solution_state = puzzle["solution_state"].split(";")
    if (sum(solution_state != state) > num_wildcards):
        print(f"Solution is incorrect for problem {id}")
        print(f"Number of wildcards: {num_wildcards}")
        print(f"Expected: \t{puzzle['solution_state']}")
        print(f"Got: \t\t{';'.join(state)}")
        print(f"Num different: {np.count_nonzero(puzzle['solution_state'].split(';') != state)}")

        for i in range(len(solution_state)):
            if solution_state[i] != state[i]:
                print(f"Sticker {i}: {state[i]} -> {solution_state[i]}")
        assert False

    if verbose:
        print("Solution is correct")

parser = argparse.ArgumentParser()
parser.add_argument("type", type=str)
parser.add_argument("--sol_dir", type=str, default="data/solutions")
parser.add_argument("--sol", type=str)
parser.add_argument("--verbose", action="store_true", default=True)
args = parser.parse_args()

args.all = None
args.lo = None
args.hi = None

if args.type == "all":
    args.all = True
elif "," in args.type:
    args.lo, args.hi = map(int, args.type.split(","))
else:
    args.id = int(args.type)

if args.all:
    args.lo = 0
    args.hi = 397

if args.lo is not None and args.hi is not None:
    for id in range(args.lo, args.hi + 1):
        sol_file_name = f"{args.sol_dir}/{id}.txt"
        if os.path.exists(sol_file_name):
            if args.verbose:
                print(f"Validating problem {id}")
            validate_one(id, sol_file_name)
else:
    assert args.id is not None

    if args.sol is None:
        args.sol = f"{args.sol_dir}/{args.id}.txt"
    validate_one(args.id, args.sol, args.verbose)
