#!/usr/bin/env python3
import argparse
import pandas as pd
from get_moves import get_moves
import os
import numpy as np

def validate_one(id, sol_file_name, verbose=False):
    puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[id]

    with open(sol_file_name, 'r') as sol_file:
        solution_state = sol_file.read()
        solution = solution_state.split('.')
        if verbose:
            print(f"Solution score: {len(solution)}")
            print(f"Solution: {solution}")
            spaced_sol = " ".join(solution[:min(50, len(solution))])
            print(spaced_sol)

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

    if (sum(puzzle["solution_state"].split(";") != state) > num_wildcards):
        print(f"Solution is incorrect for problem {id}")
        print(f"Number of wildcards: {num_wildcards}")
        print(f"Expected: {puzzle['solution_state']}")
        print(f"Got: {';'.join(state)}")
        print(f"Num different: {np.count_nonzero(puzzle['solution_state'].split(';') != state)}")
        assert False

    if verbose:
        print("Solution is correct")

parser = argparse.ArgumentParser()
parser.add_argument("--lo", type=int)
parser.add_argument("--hi", type=int)
parser.add_argument("--all", action="store_true")
parser.add_argument("--sol_dir", type=str, default="data/solutions")
parser.add_argument("--id", type=int)
parser.add_argument("--sol", type=str)
parser.add_argument("--verbose", action="store_true", default=True)
args = parser.parse_args()

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
