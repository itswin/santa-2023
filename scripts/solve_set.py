#!/usr/bin/env python3
import argparse
import pandas as pd
import itertools
import numpy as np
from subprocess import Popen, PIPE
import subprocess
from util import *
import os.path
import random

# python3 scripts/solve_set.py ID SET_NUM

def find_best_commutator(initial_state, solution_state, commutators: List[Move], solution_so_far, difference_function) -> Move:
    num_wrong = difference_function(initial_state, solution_state)

    # Go through each commutator and apply it to the initial state
    # Find the one that results in the lowest number of wrong stickers

    random.shuffle(commutators)

    # piece_to_cycle = identify_cycles(initial_state, solution_state)
    # exists_2_cycle = 2 in Counter(piece_to_cycle.values()).values()

    current_len = len(solution_so_far)

    best_pieces_solved_per_move = -1
    best_new_num_wrong = num_wrong + 5
    best_commutator = None
    for commutator_tuple in itertools.product(commutators, repeat=in_a_row):
        commutator_list = list(commutator_tuple)
        commutator = commutator_list[0]
        for i in range(1, len(commutator_list)):
            commutator = commutator.compose(commutator_list[i])

        new_state = initial_state[commutator.move]
        new_num_wrong = difference_function(new_state, solution_state)
        # if new_num_wrong < best_new_num_wrong:
        #     best_new_num_wrong = new_num_wrong
        #     best_commutator = commutator

        cancels = commutator.count_pre_cancels(solution_so_far)
        move_count = current_len + commutator.length - 2 * cancels

        # This is the inverse of the last commutator we did. Skip it.
        if move_count == 0:
            continue
        elif move_count < 0:
            print("Move count is negative. This should not happen")
            continue

        new_solved_per_move = (num_wrong - new_num_wrong) / move_count
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
parser.add_argument("set", type=int)
parser.add_argument("--partial_sol", type=str, default=None)
parser.add_argument("--manual_partial_sol", type=str, default=None)
parser.add_argument('-c','--correct', nargs='+', help='Sets already correct')

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
puzzle_type = puzzle["puzzle_type"].replace("/", "_")
print(f"puzzle_type: {puzzle_type}")

moves = get_moves(puzzle["puzzle_type"])
print(f"Number of moves: {len(moves)}")

n = int(puzzle_type.split("_")[1])

initial_state = np.array(puzzle["initial_state"].split(";"))
solution_state = np.array(puzzle["solution_state"].split(";"))

identity = np.arange(len(solution_state))

sets, piece_to_set_index, set_to_sol_piece_to_index = decompose(solution_state, moves)

if args.partial_sol:
    with open(args.partial_sol, "r") as fp:
        solution = fp.read().strip()
    for move in solution.split("."):
        initial_state = initial_state[moves[move]]
elif args.manual_partial_sol:
    solution = args.manual_partial_sol
    for move in solution.split("."):
        initial_state = initial_state[moves[move]]
else:
    if puzzle_type.startswith("cube"):
        if n % 2 == 0:
            center_orienting_seq = []
        else:
            initial_state, center_orienting_seq = orient_centers(initial_state, moves, n)

        solution = ".".join(center_orienting_seq) if center_orienting_seq else None
    else:
        solution = None

twsearch_folder = f"data/tws_phases/{puzzle_type}"
comms_folder = os.path.join(twsearch_folder, "comms")

move_map = get_move_map(n)

commutator_file = os.path.join(comms_folder, f"PIECE{args.set}.txt")
commutators = create_commutators(commutator_file, 
    moves, move_map=move_map, max_wrong=100, announce_skip=False)
print(f"Number of commutators: {len(commutators)}")

inverted_commutators = invert_moves(commutators)
commutators = commutators + inverted_commutators
print(f"Number of commutating moves after inversion: {len(commutators)}")

wildcards = puzzle['num_wildcards']

key_set = sets[args.set]
if args.correct is not None:
    for set_num in args.correct:
        key_set += sets[int(set_num)]

def difference_function(initial_state, solution_state):
    return np.count_nonzero((initial_state != solution_state)[key_set])

num_wrong = difference_function(initial_state, solution_state)
in_a_row = 1
last_num_wrong = num_wrong
iters_since_improvement = 0
best_num_wrong = num_wrong
best_solution = solution
solution_moves = []

while num_wrong > 0:
    best_comm = find_best_commutator(initial_state, solution_state, commutators, solution_moves, difference_function)

    if best_comm is None:
        print("No commutator found")
        break

    print(f"Best commutator: {best_comm.name}")
    initial_state = initial_state[best_comm.move]
    num_wrong = difference_function(initial_state, solution_state)
    print(f"\tNumber of wrong stickers: {num_wrong}")

    # print_wrong_stickers(initial_state, solution_state)

    if solution is None:
        solution = best_comm.moves_named
        solution_moves = best_comm.moves
    else:
        solution += "." + best_comm.moves_named
        solution_moves = append_move_cancel(solution_moves, best_comm)

    print(f"\tSolution length so far: {len(solution.split('.'))}")

    if num_wrong >= last_num_wrong:
        iters_since_improvement += 1
        if iters_since_improvement > 20:
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

    num_difference = difference_function(state, solution_state)
    if num_difference == 0:
        print("Solved set correctly")
        print(f"Writing to set partial solution file")

        with open(f"data/comms_partial_sol.txt", "w") as f:
            f.write(solution)
    else:
        print(f"Did not solve set correctly. {num_difference} wrong stickers")
        print(f"Writing to other partial solution file")

        with open(f"data/comms_partial_sol_wrong.txt", "w") as f:
            f.write(solution)
