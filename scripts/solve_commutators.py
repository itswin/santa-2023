#! /usr/bin/env python3

import argparse
import pandas as pd
import itertools
import numpy as np
import subprocess
from util import *
from typing import List
import random

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
        new_solved_per_move = (num_wrong - new_num_wrong) / commutator.length
        if new_num_wrong > num_wrong:
            continue
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
parser.add_argument("commutator_file", type=str)
parser.add_argument("--sol_dir", type=str, default="data/solutions")
parser.add_argument("--out_sol_dir", type=str, default="data/solutions")
parser.add_argument("--moves", action="store_true", default=False)
parser.add_argument("--unique", action="store_true", default=False)
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

commutators = create_commutators(args.commutator_file, moves, move_map, 5)
print(f"Number of commutators: {len(commutators)}")

if args.create_conjugates:
    conjugates = create_conjugates(commutators, moves, max_setup_moves=3)
    print(f"Number of conjugates: {len(conjugates)}")

    commutators = commutators + conjugates
    print(f"Number of commutating moves: {len(commutators)}")

    # Write the commutators to a file
    # with open("data/tws_phases/cube_4_4_4/expanded_comms_conjugates.txt", "w") as fp:
    #     for comm in commutators:
    #         fp.write(comm.name + "\n")

# commutators = expand_moves(commutators)
# print(f"Number of commutating moves after expansion: {len(commutators)}")


in_a_row = 1
last_num_wrong = num_wrong
iters_since_improvement = 0
while num_wrong > 0:
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

    if num_wrong == last_num_wrong:
        iters_since_improvement += 1
        if iters_since_improvement > 20:
            print("No improvement in 5 iterations. Giving up")
            break
    else:
        iters_since_improvement = 0
        last_num_wrong = num_wrong

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
#     c = 0
#     for index in wrong_indices:
#         if commutator.move[index] != index:
#             c += 1

#     assert c == np.count_nonzero(commutator.move[wrong_indices] != wrong_indices)
#     if c == 2:
#         print(f"Comm {commutator.name} affects 2 wrong stickers")
#         print(f"Move: {commutator.move}")

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

print(f"Solution length: {len(solution.split('.'))}")
print("Solution:", solution)
