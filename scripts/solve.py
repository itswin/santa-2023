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

def heuristic(state, goal_state):
    return np.count_nonzero(state != goal_state)


def improved_heuristic_with_wildcards(state, goal_state, num_wildcards):
    mismatches = np.count_nonzero(state != goal_state)
    return max(0, mismatches - num_wildcards)


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


def a_star(initial_state, goal_state, allowed_moves, num_wildcards, max_depth=30, timeout=100):
    """
    Improved A* search algorithm with wildcards, depth limit, and timeout.

    :param initial_state: List representing the initial state of the puzzle.
    :param goal_state: List representing the goal state of the puzzle.
    :param allowed_moves: Dictionary of allowed moves and their corresponding permutations.
    :param num_wildcards: Number of wildcards allowed for the puzzle.
    :param max_depth: Maximum depth to search to limit the search space.
    :param timeout: Time limit in seconds for the search.
    :return: Shortest sequence of moves to solve the puzzle, or None if no solution is found.
    """
    start_time = time.time()
    open_set = []
    heapq.heappush(open_set, Node(0, initial_state, []))  # (priority, state, path, remaining wildcards)
    closed_set = set()

    try:
        while open_set:
            current_node = heapq.heappop(open_set)

            if time.time() - start_time > timeout:
                return current_node.path, False

            if len(current_node.path) > max_depth:  # Depth limit
                continue

            if (current_node.state == goal_state).all() or improved_heuristic_with_wildcards(current_node.state, goal_state, num_wildcards) == 0:
                return current_node.path, True

            closed_set.add((tuple(current_node.state)))

            for move_name, move in allowed_moves.items():
                new_state = current_node.state[move]
                if (tuple(new_state)) not in closed_set:
                    priority = len(current_node.path) + 1 + improved_heuristic_with_wildcards(new_state, goal_state, num_wildcards)
                    heapq.heappush(open_set, Node(priority, new_state, current_node.path + [move_name]))
    except KeyboardInterrupt:
        print("Interrupted")

    return current_node.path, False  # No solution found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--max_depth", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=60 * 60 * 2)
    args = parser.parse_args()

    puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
    moves = get_moves(puzzle['puzzle_type'])

    print(puzzle)

    initial_state = np.array(puzzle["initial_state"].split(";"))
    solution_state = np.array(puzzle["solution_state"].split(";"))

    print(len(initial_state))
    print("Initial state", initial_state)
    print("Solution state", solution_state)

    current_solution = []

    with open(f"data/solutions/{args.id}.txt", "r") as fp:
        current_solution = fp.read().split(".")

    solution, valid = a_star(initial_state, solution_state, moves, puzzle["num_wildcards"], args.max_depth, args.timeout)
    if not valid:
        print("No solution found. Writing to tmp file")
        with open(f"data/a_progress/{args.id}.txt", "w") as fp:
            fp.write(".".join(solution))
    else:
        print("Solution found:", solution)
        print("Solution length:", len(solution))
        print("Current solution length:", len(current_solution))
        if len(solution) < len(current_solution):
            with open(f"data/solutions/{args.id}.txt", "w") as fp:
                fp.write(".".join(solution))
                print("Solution updated")
        else:
            print("No update")

if __name__ == "__main__":
    main()
