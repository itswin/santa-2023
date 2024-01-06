#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import heapq
import time
from typing import Dict, Tuple, List
import random
from util import *
import os

# ./scripts/ida_comms.py 240 -conjugate_file ./data/tws_phases/cube_5_5_5/expanded_comms_conjugates.txt

def evaluate_score(current_state, final_state):
    # Reward having the final position match, and also reward having 2 of the same state adjacent to each other
    # This has to be fast since it's called so often
    return np.count_nonzero(current_state != final_state)

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


def idastar(commutators, initial_state, final_state, params, current_path, clear_when_new_best):
    # Priority queue to store nodes with their f-values (g + h)
    puzzle_start_time = time.time()
    iteration_counter = 0
    current_starting_state = initial_state
    closed_set = set()

    best_state = initial_state
    best_path = current_path
    best_difference = evaluate_difference(initial_state, final_state)
    greedy = True

    try:
        while time.time() - puzzle_start_time < params['max_overall_time'] and iteration_counter < params['max_overall_iterations']:
            iteration_counter += 1

            new_state, new_path, node_counter, new_best_path, new_best_difference, new_best_state = \
                depth_limited_search(
                    commutators, current_starting_state, final_state, closed_set, params, greedy
                )

            if new_best_path is not None and new_best_difference < best_difference:
                greedy = True
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
                print("Depth limited search failed. Downsampling and increasing nodes trying again. Turning off greedy for now.")
                greedy = False
                # params['max_iteration_nodes'] = int(params['max_iteration_nodes'] * 2)

            # Downsample the closed set
            closed_set = set(random.sample(list(closed_set), int(len(closed_set)*params['set_downsampling'])))
    except KeyboardInterrupt:
        print("Keyboard Interrupt.")

    print(f"Max search time/iterations expired: {time.time() - puzzle_start_time} seconds, {iteration_counter} iterations.")
    return best_path, iteration_counter, False


def depth_limited_search(commutators, initial_state, final_state, closed_set, params, greedy):
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
        elif difference == tmp_best_difference and len(node.path) < len(tmp_best_path):
            tmp_best_state = node.state
            tmp_best_path = node.path
            tmp_best_difference = difference

        if difference <= params['wildcards']:
            # We've achieved our goal. Return the move path.
            return node.state, node.path, node_counter, tmp_best_path, tmp_best_difference, tmp_best_state

        closed_set.add(tuple(node.state))

        random.shuffle(commutators)

        for commutator in commutators:
            # Skip this move if it's the inverse of the last one we did
            # last_move = node.path[-1] if len(node.path) > 0 else None
            # if last_move is not None and \
            #         ((move_str[0] == "-" and move_str[1:] == last_move) or \
            #          (move_str[0] != "-" and "-" + move_str == last_move)):
            #     continue

            # Skip commutators which do not affect any of the wrong stickers
            wrong_indices = np.where(final_state != node.state)[0]
            num_affected = np.count_nonzero(commutator.move[wrong_indices] != wrong_indices)
            if num_affected == 0:
                continue

            cancels = commutator.count_pre_cancels(node.path)
            new_state = node.state[commutator.move]
            new_difference = evaluate_difference(new_state, final_state)

            if new_difference >= difference and greedy:
                continue

            if cancels:
                new_path = node.path[:-cancels] + commutator.moves[cancels:]
            else:
                new_path = node.path + commutator.moves

            if tuple(new_state) not in closed_set:
                heapq.heappush(open_set, Node(node.priority + commutator.length - 2 * cancels + new_difference, new_state, new_path))
                node_counter += 1

    # If no solutions are found:
    print("Open set completed. No solutions.")
    return None, None, node_counter, tmp_best_path, tmp_best_difference, tmp_best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=int)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-commutator_file", type=str)
    group.add_argument("-conjugate_file", type=str)

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
    parser.add_argument("--partial_sol", type=str, default=None)
    parser.add_argument("--add_normal_moves", action="store_true", default=False)
    args = parser.parse_args()

    puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
    puzzle_type = puzzle["puzzle_type"]
    moves = get_moves(puzzle_type)

    print(puzzle)

    initial_state = np.array(puzzle["initial_state"].split(";"))
    solution_state = np.array(puzzle["solution_state"].split(";"))

    n = int(puzzle_type.split("/")[-1])
    move_map = get_move_map(n)

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
            conjugate_file = os.path.join(commutator_folder, "expanded_comms_conjugates2.txt")
            with open(conjugate_file, "w") as fp:
                for comm in commutators:
                    fp.write(comm.name + "\n")
    elif args.conjugate_file:
        commutators = read_conjugates(args.conjugate_file, moves)
        print(f"Number of conjugates: {len(commutators)}")

    wildcards = puzzle['num_wildcards']
    current_solution = []

    with open(f"{args.sol_dir}/{args.id}.txt", "r") as fp:
        current_solution = fp.read().split(".")

    # Add normal moves to the commutators
    if args.add_normal_moves:
        for name, move in moves.items():
            commutators.append(Move(name, move, [name]))

    params = {
        'wildcards': wildcards,
        'max_iteration_time': args.max_iter_time,
        'max_iteration_nodes': args.max_iter_nodes,
        'max_overall_time': args.timeout,
        'max_overall_iterations': args.iterations,
        'max_moves': len(current_solution),
        'set_downsampling': args.downsampling
    }

    progress = []
    if args.partial_sol:
        print("Starting with partial solution")
        with open(args.partial_sol, "r") as fp:
            progress = fp.read().split(".")
        for move in progress:
            initial_state = initial_state[moves[move]]
        print(f"Partial sol length: {len(progress)}. Diff: {evaluate_difference(initial_state, solution_state)}")
    elif args.from_progress:
        print("Picking up from progress")
        with open(f"data/ida_progress/{args.id}.txt", "r") as fp:
            progress = fp.read().split(".")
        for move in progress:
            initial_state = initial_state[moves[move]]
        print(f"Progress length: {len(progress)}. Diff: {evaluate_difference(initial_state, solution_state)}")

    print(f"Starting testing with parameters: {params}")
    solution_path, iteration_counter, valid = idastar(commutators, initial_state, solution_state, params, progress, args.clear_when_new_best)
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
