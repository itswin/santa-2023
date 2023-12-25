#!/usr/bin/env python3
import pandas as pd
import json
import numpy as np
import argparse
import heapq
import time
from typing import Dict, Tuple, List
import sys
from get_moves import get_moves
import random

def evaluate_difference(current_state, final_state):
    return np.count_nonzero(current_state != final_state)

def evaluate_score(current_state, final_state):
    # Reward having the final position match, and also reward having 2 of the same state adjacent to each other
    # This has to be fast since it's called so often
    return np.count_nonzero(current_state != final_state) + \
        np.count_nonzero(current_state[1:] != current_state[:-1]) + \
        0.5 * np.count_nonzero(current_state[2:] != current_state[:-2])

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

    best_state = None
    best_path = None
    best_difference = len(initial_state)

    try:
        while time.time() - puzzle_start_time < params['max_overall_time'] and iteration_counter < params['max_overall_iterations']:
            iteration_counter += 1

            new_state, new_path, node_counter, new_best_path, new_best_difference, new_best_state = depth_limited_search(move_dict, current_starting_state, final_state, closed_set, params)

            if new_best_path is not None and new_best_difference < best_difference:
                best_state = new_best_state
                best_path = current_path + new_best_path
                best_difference = new_best_difference
                current_path = best_path
                print(f"New best path found: {best_path}")
                print(f"Difference {best_difference}")
                if clear_when_new_best:
                    print(f"Iteration #{iteration_counter} completed. Nodes: {node_counter}. Clearing closed set.")
                    closed_set = set()
                    current_starting_state = best_state
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
    best_state = None
    best_path = None
    best_difference = len(initial_state)

    while open_set:
        # Get the node with the lowest f-value
        node = heapq.heappop(open_set)

        # Check for timeout
        if time.time() - start_time > params['max_iteration_time']:
#             print("Iteration Timed Out.")
            return node.state, node.path, node_counter, best_path, best_difference, best_state

        if node_counter > params['max_iteration_nodes']:
#             print("Iteration Node Limit Reached.")
            return node.state, node.path, node_counter, best_path, best_difference, best_state

        difference = evaluate_difference(node.state, final_state)
    
        if difference < best_difference:
            best_state = node.state
            best_path = node.path
            best_difference = difference

        if difference <= params['wildcards']:
            # We've achieved our goal. Return the move path.
            return node.state, node.path, node_counter, best_path, best_difference, best_state

        closed_set.add(tuple(node.state))

        for move_str, move in move_dict.items():
            new_state = node.state[move]
            if tuple(new_state) not in closed_set:
                heapq.heappush(open_set, Node(len(node.path) + 1 + evaluate_score(new_state, final_state), new_state, node.path + [move_str]))
                node_counter += 1

    # If no solutions are found:
    print("Open set completed. No solutions.")
    return None, None, node_counter, best_path, best_difference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=60 * 60 * 2)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--from_progress", action="store_true", default=False)
    parser.add_argument("--max_iter_time", type=int, default=30)
    parser.add_argument("--max_iter_nodes", type=int, default=500000)
    parser.add_argument("--clear_when_new_best", action="store_true", default=False)
    args = parser.parse_args()

    puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
    moves = get_moves(puzzle['puzzle_type'])

    print(puzzle)

    initial_state = np.array(puzzle["initial_state"].split(";"))
    solution_state = np.array(puzzle["solution_state"].split(";"))

    moves = get_moves(puzzle['puzzle_type'])

    wildcards = puzzle['num_wildcards']
    current_solution = []

    with open(f"data/solutions/{args.id}.txt", "r") as fp:
        current_solution = fp.read().split(".")

    params = {
        'wildcards': wildcards,
        'max_iteration_time': args.max_iter_time,
        'max_iteration_nodes': args.max_iter_nodes,
        'max_overall_time': args.timeout,
        'max_overall_iterations': args.iterations,
        'max_moves': len(current_solution),
        'set_downsampling': 0.8
    }

    progress = []
    if args.from_progress:
        print("Picking up from progress")
        with open(f"data/ida_progress/{args.id}.txt", "r") as fp:
            progress = fp.read().split(".")
        for move in progress:
            initial_state = initial_state[moves[move]]
        print(f"Progress length: {len(progress)}. Diff: {evaluate_difference(initial_state, solution_state)}")

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
            with open(f"data/solutions/{args.id}.txt", "w") as fp:
                fp.write(".".join(solution_path))
    else:
        print(f"No solution found in {iteration_counter} iterations.")
        print("Writing to tmp file.")

        print(f"Length of best path: {len(solution_path)}")
        print(f"Best path: {solution_path}")

        with open(f"data/ida_progress/{args.id}.txt", "w") as fp:
            fp.write(".".join(solution_path))

if __name__ == "__main__":
    main()
