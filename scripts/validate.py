#!/usr/bin/env python3
import argparse
import pandas as pd
from get_moves import get_moves
import os

def validate_one(problem_id, sol_file_name, verbose=False):
    puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[problem_id]

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

    state = puzzle["initial_state"].split(";")
    if verbose:
        print(f"Initial state: {state}")
        print(f"Solution state: {puzzle['solution_state']}")
    for move_name in solution:
        state = [state[i] for i in moves[move_name]]
    if (puzzle["solution_state"].split(";") != state):
        print(f"Solution is incorrect for problem {problem_id}")
        print(f"Expected: {puzzle['solution_state']}")
        print(f"Got: {';'.join(state)}")
        assert False

    if verbose:
        print("Solution is correct")

parser = argparse.ArgumentParser()
parser.add_argument("--all", action="store_true")
parser.add_argument("--sol_dir", type=str, default="data/solutions")
parser.add_argument("--problem_id", type=int)
parser.add_argument("--sol", type=str)
parser.add_argument("--verbose", action="store_true", default=True)
args = parser.parse_args()

if args.all:
    for problem_id in range(1, 397):
        sol_file_name = f"{args.sol_dir}/{problem_id}.txt"
        if os.path.exists(sol_file_name):
            if args.verbose:
                print(f"Validating problem {problem_id}")
            validate_one(problem_id, sol_file_name)
else:
    assert args.problem_id is not None

    if args.sol is None:
        args.sol = f"{args.sol_dir}/{args.problem_id}.txt"
    validate_one(args.problem_id, args.sol, args.verbose)
