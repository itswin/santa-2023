#!/usr/bin/env python3
import pandas as pd
import json
import numpy as np
import argparse
import heapq
import time
from typing import Dict, Tuple, List
import sys
import random
from util import *

def evaluate_score(current_state, final_state):
    # Reward having the final position match, and also reward having 2 of the same state adjacent to each other
    # This has to be fast since it's called so often
    score = np.count_nonzero(current_state != final_state)
        # np.count_nonzero(current_state[25:] != current_state[:-25])
        # np.count_nonzero(current_state[2:] != current_state[:-2])
        # np.count_nonzero(current_state[3:] != current_state[:-3]) + \
        # np.count_nonzero(current_state[4:] != current_state[:-4])
    
    # Circle one is 0-99. We want 0 to be the same as 25, 1 to be the same as 26, etc. modulo 100
    # for i in range(100):
    #     if current_state[i] != current_state[(i+25)%100]:
    #         score += 1
    # for i in range(100):
    #     if current_state[i] != current_state[((i+73)%100 + 100) % 198]:
    #         score += 1

    # Additionally, count the number of "segments" which are the same consecutive piece
    piece = current_state[0]
    num_segments = 1
    segment_length = 1
    for i in range(1, len(current_state)):
        if current_state[i] != piece:
            piece = current_state[i]
            num_segments += 1
            score += 3 / segment_length
            segment_length = 1
        else:
            segment_length += 1

    return score + num_segments

# def evaluate_score(current_state, final_state):
#     wrong_indices = [0, 25]
#     score = normal_evaluate_score(current_state, final_state)

#     # Also add cost for distance of wrong stickers from "wrong_indices"
#     for i, piece in enumerate(current_state):
#         if piece != final_state[i]:
#             score += min(abs(i - wrong_indices[0]), abs(i - wrong_indices[1])) / 25

#     return score

# def evaluate_score(current_state, final_state):
#     score = np.count_nonzero(current_state != final_state)
#     for piece in "ABCDEFGHIJKLMNOP":
#         index = np.where(current_state == piece)[0][0]
#         score += lookup_table[piece][index]
#     return score

class Node:
    def __init__(self, priority, state, path):
        self.priority = priority
        self.state = state
        self.path = path

    def __lt__(self, other):
        return self.priority < other.priority
    
    def __gt__(self, other):
        return self.priority > other.priority

    def __repr__(self):
        return f"Node({self.state}, {self.priority}, {self.path})"


def idastar(move_dict, initial_state, final_state, params, current_path=[], clear_when_new_best=False):
    # Priority queue to store nodes with their f-values (g + h)
    puzzle_start_time = time.time()
    iteration_counter = 0
    current_starting_state = initial_state
    closed_set = set()

    best_state = initial_state
    best_path = current_path
    best_difference = evaluate_difference(initial_state, final_state)

    try:
        while time.time() - puzzle_start_time < params['max_overall_time'] and iteration_counter < params['max_overall_iterations']:
            iteration_counter += 1

            new_state, new_path, node_counter, new_best_path, new_best_difference, new_best_state = depth_limited_search(move_dict, current_starting_state, final_state, closed_set, params)

            if new_best_path is not None and new_best_difference < best_difference:
                best_state = new_best_state
                best_path = current_path.copy() + new_best_path
                best_difference = new_best_difference
                print(f"New best path found: {best_path}")
                print(f"Difference {best_difference}. Length: {len(best_path)}")
                if clear_when_new_best:
                    print(f"Iteration #{iteration_counter} completed. Nodes: {node_counter}. Clearing closed set.")
                    closed_set = set()
                    current_starting_state = best_state
                    current_path = best_path.copy()
                    continue

            if new_state is not None:
                current_path += new_path
                current_difference = evaluate_difference(new_state, final_state)
                current_score = evaluate_score(new_state, final_state)

                print(f"Iteration #{iteration_counter} completed - New Moves: {len(new_path)}, Total Moves: {len(current_path)}, Difference: {current_difference}, Score: {current_score}, Nodes: {node_counter}")
                current_starting_state = new_state

                if current_difference <= params['wildcards']:
                    # We've achieved our goal. Return the move path.
                    return current_path, iteration_counter, True
                if len(current_path) >= params['max_moves']:
                    print(f"Max moves reached. Returning best path.")
                    return best_path, iteration_counter, False
            else:
                print("Depth limited search failed. Downsampling and increasing nodes trying again.")
                # params['max_iteration_nodes'] = int(params['max_iteration_nodes'] * 2)

            # Downsample the closed set
            closed_set = set(random.sample(list(closed_set), int(len(closed_set)*params['set_downsampling'])))
    except KeyboardInterrupt:
        print("Keyboard Interrupt.")

    print(f"Max search time/iterations expired: {time.time() - puzzle_start_time} seconds, {iteration_counter} iterations.")
    return best_path, iteration_counter, False


def depth_limited_search(move_dict, initial_state, final_state, closed_set, params):
    # Priority queue to store nodes with their f-values (g + h)
    start_time = time.time()
    open_set = []
    node_counter = 1

    heapq.heappush(open_set, Node(0, initial_state, []))  # (priority, state, path)
    # Set to keep track of visited nodes
    tmp_best_state = None
    tmp_best_path = None
    tmp_best_difference = len(initial_state)

    while open_set:
        # Get the node with the lowest f-value
        node = heapq.heappop(open_set)

        # Check for timeout
        if time.time() - start_time > params['max_iteration_time']:
#             print("Iteration Timed Out.")
            return node.state, node.path, node_counter, tmp_best_path, tmp_best_difference, tmp_best_state

        if node_counter > params['max_iteration_nodes']:
#             print("Iteration Node Limit Reached.")
            return node.state, node.path, node_counter, tmp_best_path, tmp_best_difference, tmp_best_state

        difference = evaluate_difference(node.state, final_state)
    
        if difference < tmp_best_difference:
            tmp_best_state = node.state
            tmp_best_path = node.path
            tmp_best_difference = difference

        if difference <= params['wildcards']:
            # We've achieved our goal. Return the move path.
            return node.state, node.path, node_counter, tmp_best_path, tmp_best_difference, tmp_best_state

        closed_set.add(tuple(node.state))

        for move_str, move in move_dict.items():
            move_dot_pos = move_str.find(".")
            move_type = remove_invert(move_str if move_dot_pos == -1 else move_str[:move_dot_pos])

            # Skip this move if it's the inverse of the last one we did
            last_move = node.path[-1] if len(node.path) > 0 else None
            if last_move is not None:
                last_dot_pos = last_move.find(".")
                last_move_type = remove_invert(last_move if last_dot_pos == -1 else last_move[:last_dot_pos])
                if move_type == last_move_type:
                    continue

            new_state = node.state[move]
            if tuple(new_state) not in closed_set:
                move_expanded = move_str.split(".")
                heapq.heappush(open_set, Node(len(node.path) + len(move_expanded) / 10 + evaluate_score(new_state, final_state), new_state, node.path + move_expanded))
                node_counter += 1

    # If no solutions are found:
    print("Open set completed. No solutions.")
    return None, None, node_counter, tmp_best_path, tmp_best_difference, tmp_best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=60 * 60 * 2)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--from_progress", action="store_true", default=False)
    parser.add_argument("--max_iter_time", type=int, default=30)
    parser.add_argument("--max_iter_nodes", type=int, default=500000)
    parser.add_argument("--clear_when_new_best", action="store_true", default=False)
    parser.add_argument("--downsampling", type=float, default=0.8)
    parser.add_argument("--sol_dir", type=str, default="data/solutions")
    parser.add_argument("--out_sol_dir", type=str, default="data/solutions")
    parser.add_argument("--always_write", action="store_true", default=False)
    parser.add_argument("--max_moves", type=int, default=None)
    args = parser.parse_args()

    puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
    moves = get_moves(puzzle['puzzle_type'])

    print(puzzle)

    initial_state = np.array(puzzle["initial_state"].split(";"))
    solution_state = np.array(puzzle["solution_state"].split(";"))

    difference = evaluate_difference(initial_state, solution_state)
    print(f"Initial difference: {difference}")

    moves = get_moves(puzzle['puzzle_type'])
    moves = expand_moves(moves)
    # print(moves)

    wildcards = puzzle['num_wildcards']
    current_solution = []

    with open(f"{args.sol_dir}/{args.id}.txt", "r") as fp:
        current_solution = fp.read().split(".")

    params = {
        'wildcards': wildcards,
        'max_iteration_time': args.max_iter_time,
        'max_iteration_nodes': args.max_iter_nodes,
        'max_overall_time': args.timeout,
        'max_overall_iterations': args.iterations,
        'max_moves': len(current_solution) if args.max_moves is None else args.max_moves,
        'set_downsampling': args.downsampling
    }

    progress = []
    if args.from_progress:
        print("Picking up from progress")
        with open(f"data/ida_progress/{args.id}.txt", "r") as fp:
            progress = fp.read().split(".")
        for move in progress:
            initial_state = initial_state[moves[move]]
        print(f"Progress length: {len(progress)}. Diff: {evaluate_difference(initial_state, solution_state)}")

    # fake_initial_state = initial_state.copy()
    # fake_initial_state[fake_initial_state == "C"] = "A"
    # fake_solution_state = solution_state.copy()
    # fake_solution_state[fake_solution_state == "C"] = "A"
    # print(f"Fake initial state: {fake_initial_state}")
    # print(f"Fake solution state: {fake_solution_state}")

    print(f"Starting testing with parameters: {params}")
    solution_path, iteration_counter, valid = idastar(moves, initial_state, solution_state, params, progress, args.clear_when_new_best)
    if valid:
        print(f"Solution found in {iteration_counter} iterations.")
        print(f"Solution path: {solution_path}")
        print(f"Solution length: {len(solution_path)}")

        print(f"Validating")
        state = np.array(puzzle["initial_state"].split(";"))
        for move_name in solution_path:
            state = state[moves[move_name]]

        differences = evaluate_difference(state, solution_state)
        if differences <= wildcards:
            print(f"Solution is valid. Diff to WC: {differences} <= {wildcards}")
        else:
            print("Solution is invalid")
            print(f"Expected: {solution_state}")
            print(f"Got: {state}")
            assert False

        if len(solution_path) < len(current_solution):
            print(f"New solution is shorter than current solution. Writing to file.")
            with open(f"{args.out_sol_dir}/{args.id}.txt", "w") as fp:
                fp.write(".".join(solution_path))
        elif args.always_write:
            print(f"New solution is longer than current solution. Writing to file.")
            with open(f"{args.out_sol_dir}/{args.id}_other.txt", "w") as fp:
                fp.write(".".join(solution_path))
    else:
        print(f"No solution found in {iteration_counter} iterations.")
        print("Writing to tmp file.")

        state = np.array(puzzle["initial_state"].split(";"))
        for move_name in solution_path:
            state = state[moves[move_name]]

        differences = evaluate_difference(state, solution_state)

        print(f"Length of best path: {len(solution_path)}")
        print(f"Best path: {solution_path}")
        print(f"Best difference: {differences}")

        with open(f"data/ida_progress/{args.id}.txt", "w") as fp:
            fp.write(".".join(solution_path))

if __name__ == "__main__":
    main()
