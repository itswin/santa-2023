#! /usr/bin/env python3

import argparse
import pandas as pd
import itertools
import numpy as np
import subprocess
from util import *
from typing import List
import random
import os

# seq 209 209 | xargs -I {} sh -c "python3 scripts/solve_reskin.py {} && ./scripts/ida_comms.py {} ./data/tws_phases/cube_4_4_4/4x4x4_center_comms.txt --partial_sol data/partial_sol.txt --clear_when_new_best"

def find_best_commutator(initial_state, solution_state, commutators: List[Move], in_a_row=1) -> Move:
    num_wrong = evaluate_difference(initial_state, solution_state)

    # Go through each commutator and apply it to the initial state
    # Find the one that results in the lowest number of wrong stickers

    random.shuffle(commutators)

    best_pieces_solved_per_move = -1
    best_new_num_wrong = num_wrong + 5
    best_commutator = None
    for commutator_tuple in itertools.product(commutators, repeat=in_a_row):
        commutator_list = list(commutator_tuple)
        commutator = commutator_list[0]
        for i in range(1, len(commutator_list)):
            commutator = commutator.compose(commutator_list[i])

        new_state = initial_state[commutator.move]
        new_num_wrong = evaluate_difference(new_state, solution_state)
        if new_num_wrong > num_wrong:
            continue

        new_solved_per_move = (num_wrong - new_num_wrong) / commutator.length
        if new_solved_per_move > best_pieces_solved_per_move:
            best_pieces_solved_per_move = new_solved_per_move
            best_commutator = commutator

    # if best_new_num_wrong >= num_wrong:
    #     print(f"Cound not find a commutator that improved the number of wrong stickers. Best stickers {best_new_num_wrong} vs {num_wrong}")
    #     print(f"Best commutator: {best_commutator.name} ({best_commutator.move})")
    #     return None

    return best_commutator

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-commutator_file", type=str)
group.add_argument("-conjugate_file", type=str)

parser.add_argument("--out_sol_dir", type=str, default="data/solutions")
parser.add_argument("--partial_sol", type=str, default=None)
parser.add_argument("--create_conjugates", action="store_true", default=False)

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

if puzzle_type.startswith("cube"):
    move_map = get_move_map(n)
else:
    move_map = None

num_wrong = evaluate_difference(initial_state, solution_state)
print(f"Number of wrong stickers: {num_wrong}")

if args.commutator_file:
    commutators = create_commutators(args.commutator_file, moves, move_map, 5)
    print(f"Number of commutators: {len(commutators)}")

    if args.create_conjugates:
        conjugates = create_conjugates(commutators, moves, max_setup_moves=2)
        print(f"Number of conjugates: {len(conjugates)}")

        commutators = commutators + conjugates
        print(f"Number of commutating moves: {len(commutators)}")

        # Write the commutators to a file in the same folder as the commutator file
        commutator_folder = os.path.dirname(args.commutator_file)  
        conjugate_file = os.path.join(commutator_folder, "expanded_comms_conjugates.txt")
        with open(conjugate_file, "w") as fp:
            for comm in commutators:
                fp.write(comm.name + "\n")

elif args.conjugate_file:
    commutators = read_conjugates(args.conjugate_file, moves)
    print(f"Number of conjugates: {len(commutators)}")

inverted_commutators = invert_moves(commutators)
commutators = commutators + inverted_commutators
print(f"Number of commutating moves after inversion: {len(commutators)}")

wildcards = puzzle['num_wildcards']

in_a_row = 1
last_num_wrong = num_wrong
iters_since_improvement = 0
best_num_wrong = num_wrong
best_solution = None

while num_wrong > wildcards:
    best_comm = find_best_commutator(initial_state, solution_state, commutators, in_a_row)

    if best_comm is None:
        print("No commutator found")
        break
        # in_a_row += 1
        # if in_a_row > 2:
        #     print("Giving up")
        #     break
        # print("No commutator found. Trying more in a row")
        # print(f"Current state: {initial_state}")
        # continue

    print(f"Best commutator: {best_comm.name} ({best_comm.move})")
    initial_state = initial_state[best_comm.move]
    num_wrong = evaluate_difference(initial_state, solution_state)
    print(f"Number of wrong stickers: {num_wrong}")

    if solution is None:
        solution = best_comm.moves_named
    else:
        solution += "." + best_comm.moves_named

    print(f"Solution length so far: {len(solution.split('.'))}")

    if num_wrong >= last_num_wrong:
        iters_since_improvement += 1
        if iters_since_improvement > 5:
            print("No improvement in 5 iterations. Giving up")
            break
    else:
        iters_since_improvement = 0
        last_num_wrong = num_wrong

    if num_wrong < best_num_wrong:
        best_num_wrong = num_wrong
        best_solution = solution

solution_moves = solution.split(".")
print(f"Last Solution length: {len(solution_moves)}")
print("Solution:", solution)

best_solution_moves = best_solution.split(".")
print(f"Best Solution length: {len(best_solution_moves)}")
print("Best Solution:", best_solution)

with open(f"data/solutions/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

print(f"Validating")
state = np.array(puzzle["initial_state"].split(";"))
for move_name in best_solution_moves:
    state = state[moves[move_name]]

num_difference = evaluate_difference(state, solution_state)

if num_difference <= wildcards:
    print(f"Solution is valid. Diff to WC: {num_difference} <= {wildcards}")
    # Write it to the solution file

    print(f"Length of new solution: {len(best_solution_moves)}")
    print(f"Length of current solution: {len(current_solution)}")
    if len(best_solution_moves) < len(current_solution):
        print(f"New solution is shorter than current solution. Writing to file.")
        with open(f"{args.out_sol_dir}/{args.id}.txt", "w") as fp:
            fp.write(solution)
    else:
        print(f"New solution is longer than current solution.")
else:
    print(f"Solution is invalid. Diff to WC: {num_difference} > {wildcards}")
    print(f"Expected: {solution_state}")
    print(f"Got: {state}")
    print(f"Writing to partial solution file")

    with open(f"data/comms_partial_sol.txt", "w") as f:
        f.write(solution)
