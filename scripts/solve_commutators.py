#! /usr/bin/env python3

import argparse
from collections import Counter
import pandas as pd
import itertools
import numpy as np
import subprocess
from util import *
from typing import List
import random
import os

# seq 209 209 | xargs -I {} sh -c "python3 scripts/solve_reskin.py {} && ./scripts/ida_comms.py {} ./data/tws_phases/cube_4_4_4/4x4x4_center_comms.txt --partial_sol data/partial_sol.txt --clear_when_new_best"

# Identify cycles and find a commutator that affects them both
def join_cycles(initial_state, solution_state, commutators: List[Move]) -> Move:
    piece_to_cycle = identify_cycles(initial_state, solution_state)

    print(f"Found {len(set(piece_to_cycle.values()))} cycles")
    print(f"Piece to cycle: {piece_to_cycle}")

    # Find a commutator that affects both cycles that results in the fewest number of cycles
    best_commutator = None
    exists_2_cycle = 2 in Counter(piece_to_cycle.values()).values()
    num_cycles = len(set(piece_to_cycle.values()))
    best_compare = (exists_2_cycle, num_cycles + 5)

    for commutator in commutators:
        # Check if this commutator affects both cycles
        if not any(commutator.move[i] != i for i in piece_to_cycle.keys()):
            continue

        # Apply the commutator to the initial state
        new_state = initial_state[commutator.move]

        # Identify the cycles in the new state
        new_piece_to_cycle = identify_cycles(new_state, solution_state)
        num_cycles = len(set(new_piece_to_cycle.values()))
        new_exists_2_cycle = 2 in Counter(new_piece_to_cycle.values()).values()
        compare = (new_exists_2_cycle, num_cycles)

        if compare < best_compare:
            best_compare = compare
            best_commutator = commutator

    return best_commutator

def remove_2_cycles(state, solution_state, commutators: List[Move]):
    piece_to_cycle = identify_cycles(state, solution_state)

    # print(f"Found {len(set(piece_to_cycle.values()))} cycles")
    # print(f"Piece to cycle: {piece_to_cycle}")

    # Find a commutator that affects both cycles that results in the fewest number of cycles
    result_commutator = None

    while 2 in Counter(piece_to_cycle.values()).values():
        join_commutator = join_cycles(state, solution_state, commutators)
        if join_commutator is None:
            print("No join commutator found to remove 2 cycles")
            break

        # print(f"Join commutator: {join_commutator.name}")
        state = state[join_commutator.move]
        piece_to_cycle = identify_cycles(state, solution_state)

        if result_commutator is None:
            result_commutator = join_commutator
        else:
            result_commutator = result_commutator.compose(join_commutator)

    return result_commutator

def make_center_evalute_difference(n):
    center_indices = get_centers(n)
    return lambda state, solution_state: np.count_nonzero(state[center_indices] != solution_state[center_indices])

def find_best_commutator(initial_state, solution_state, commutators: List[Move], in_a_row=1, difference_function=evaluate_difference) -> Move:
    num_wrong = difference_function(initial_state, solution_state)

    # Go through each commutator and apply it to the initial state
    # Find the one that results in the lowest number of wrong stickers

    random.shuffle(commutators)

    # piece_to_cycle = identify_cycles(initial_state, solution_state)
    # exists_2_cycle = 2 in Counter(piece_to_cycle.values()).values()

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
        # if new_num_wrong > num_wrong:
        #     continue

        # if new_num_wrong == num_wrong:
        #     continue

        # if new_num_wrong == 4:
        #     continue

        # Do not create a 2 cycle if there is not already one
        # if not exists_2_cycle:
        #     new_piece_to_cycle = identify_cycles(new_state, solution_state)
        #     new_exists_2_cycle = 2 in Counter(new_piece_to_cycle.values()).values()
        #     if new_exists_2_cycle:
        #         continue

        # if new_num_wrong < best_new_num_wrong:
        #     best_new_num_wrong = new_num_wrong
        #     best_commutator = commutator

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
parser.add_argument("--centers_only", action="store_true", default=False)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
print(puzzle)

puzzle_type = puzzle["puzzle_type"]
n = int(puzzle_type.split("/")[-1])
moves = get_moves(puzzle["puzzle_type"])
print(f"Number of moves: {len(moves)}")

if args.centers_only:
    assert puzzle_type.startswith("cube")
    difference_function = make_center_evalute_difference(n)
else:
    difference_function = evaluate_difference

initial_state = np.array(puzzle["initial_state"].split(";"))
solution_state = np.array(puzzle["solution_state"].split(";"))

if args.partial_sol:
    with open(args.partial_sol, "r") as fp:
        solution = fp.read().strip()
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

if puzzle_type.startswith("cube"):
    move_map = get_move_map(n)
    print(move_map)
else:
    move_map = None

num_wrong = evaluate_difference(initial_state, solution_state)
print(f"Number of wrong stickers: {num_wrong}")

if args.commutator_file:
    commutators = create_commutators(args.commutator_file, moves, move_map, 25, False)
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
    commutators = read_conjugates(args.conjugate_file, moves, None, 25, False)
    print(f"Number of conjugates: {len(commutators)}")

inverted_commutators = invert_moves(commutators)
commutators = commutators + inverted_commutators
print(f"Number of commutating moves after inversion: {len(commutators)}")

wildcards = puzzle['num_wildcards']

in_a_row = 1
last_num_wrong = num_wrong
iters_since_improvement = 0
best_num_wrong = num_wrong
best_solution = solution

while num_wrong > wildcards:
    best_comm = find_best_commutator(initial_state, solution_state, commutators, in_a_row, difference_function)

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

    print(f"Best commutator: {best_comm.name}")
    initial_state = initial_state[best_comm.move]
    num_wrong = difference_function(initial_state, solution_state)
    print(f"\tNumber of wrong stickers: {num_wrong}")

    # print_wrong_stickers(initial_state, solution_state)

    if solution is None:
        solution = best_comm.moves_named
    else:
        solution += "." + best_comm.moves_named

    print(f"\tSolution length so far: {len(solution.split('.'))}")

    if num_wrong >= last_num_wrong:
        iters_since_improvement += 1
        if iters_since_improvement > 20:
            print("No improvement in 5 iterations. Giving up")
            break

        # join_commutator = remove_2_cycles(initial_state, solution_state, commutators)
        # if join_commutator is None:
        #     print("No 2 cycles")
        #     continue

        # print(f"Join commutator: {join_commutator.name}")
        # initial_state = initial_state[join_commutator.move]
        # num_wrong = difference_function(initial_state, solution_state)
        # last_num_wrong = num_wrong
        # print(f"\tNumber of wrong stickers: {num_wrong}")

        # # print_wrong_stickers(initial_state, solution_state)

        # if solution is None:
        #     solution = join_commutator.moves_named
        # else:
        #     solution += "." + join_commutator.moves_named
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


# for commutator in commutators:
#     if commutator.move[7] != 7:
#         print(f"Comm {commutator.name} affects 7. Num wrong {commutator.num_wrong}")
#         print(f"Move: {commutator.move}")
#         break
# wrong_indices = np.where(solution_state != initial_state)[0]
# print(sorted(wrong_indices))
# wrong_to_count = {}
# for commutator in commutators:
#     # Count how many of the wrong stickers are affected by this commutator
#     c = np.count_nonzero(commutator.move[wrong_indices] != wrong_indices)
#     wrong_to_count[c] = wrong_to_count.get(c, 0) + 1

# print("Wrong to count", wrong_to_count)

# index_affected_count = {}
# for commutator in commutators:
#     for i in range(len(commutator.move)):
#         if commutator.move[i] != i:
#             index_affected_count[i] = index_affected_count.get(i, 0) + 1

# print("Index affected count:")
# for index, count in sorted(index_affected_count.items()):
#     print(f"{index}: {count}")

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
