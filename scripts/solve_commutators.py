#! /usr/bin/env python3

import argparse
import pandas as pd
import itertools
import numpy as np
import subprocess
from util import *

# seq 209 209 | xargs -I {} sh -c "python3 scripts/solve_reskin.py {} && ./scripts/ida_comms.py {} ./data/tws_phases/cube_4_4_4/4x4x4_center_comms.txt --partial_sol data/partial_sol.txt --clear_when_new_best"

def find_best_commutator(initial_state, solution_state, commutators) -> Commutator:
    num_wrong = evaluate_difference(initial_state, solution_state)

    # Go through each commutator and apply it to the initial state
    # Find the one that results in the lowest number of wrong stickers

    best_pieces_solved_per_move = 0
    best_commutator = None
    for commutator in commutators:
        new_state = initial_state[commutator.move]
        new_num_wrong = evaluate_difference(new_state, solution_state)
        new_solved_per_move = (num_wrong - new_num_wrong) / commutator.length
        if new_solved_per_move > 0:
            print(f"{commutator.name} ({commutator.move}): {new_num_wrong} wrong, {new_solved_per_move} solved per move")
            if new_solved_per_move > best_pieces_solved_per_move:
                best_pieces_solved_per_move = new_num_wrong
                best_commutator = commutator

    return best_commutator

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("commutator_file", type=str)
parser.add_argument("--sol_dir", type=str, default="data/solutions")
parser.add_argument("--out_sol_dir", type=str, default="data/solutions")
parser.add_argument("--moves", action="store_true", default=False)
parser.add_argument("--unique", action="store_true", default=False)
parser.add_argument("--partial_sol", type=str, default=None)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
print(puzzle)

puzzle_type = puzzle["puzzle_type"]
n = int(puzzle_type.split("/")[-1])
moves = get_moves(puzzle["puzzle_type"])
print(f"Number of moves: {len(moves)}")

initial_state = np.array(puzzle["initial_state"].split(";"))
solution_state = np.array(puzzle["solution_state"].split(";"))


if args.partial_sol:
    with open(args.partial_sol, "r") as fp:
        solution = fp.read().strip()
    for move in solution.split("."):
        initial_state = initial_state[moves[move]]
else:
    solution = None

move_map = get_move_map(n)

commutators = create_commutators(args.commutator_file, moves, move_map)
print(f"Number of commutators: {len(commutators)}")

num_wrong = evaluate_difference(initial_state, solution_state)
print(f"Number of wrong stickers: {num_wrong}")

while num_wrong > 0:
    best_comm = find_best_commutator(initial_state, solution_state, commutators)

    if best_comm is None:
        print("No commutator found")
        break

    print(f"Best commutator: {best_comm.name} ({best_comm.move})")
    initial_state = initial_state[best_comm.move]
    num_wrong = evaluate_difference(initial_state, solution_state)
    print(f"Number of wrong stickers: {num_wrong}")

    if solution is None:
        solution = best_comm.moves_named
    else:
        solution += "." + best_comm.moves_named
    exit()

print(f"Solution length: {len(solution.split('.'))}")
print("Solution:", solution)
