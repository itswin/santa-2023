#!/usr/bin/env python3
import argparse
import pandas as pd
import itertools
import numpy as np
import subprocess
from util import *

# seq 210 234 | xargs -P 4 -I {} python3 scripts/solve_reskin.py {}

def pseudo(id):
    puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[id]
    print(puzzle)

    puzzle_type = puzzle["puzzle_type"]
    n = int(puzzle_type.split("/")[-1])
    moves = get_moves(puzzle["puzzle_type"])
    print(f"Number of moves: {len(moves)}")

    initial_state = puzzle["initial_state"].split(";")
    solution_state = puzzle["solution_state"].split(";")

    STICKER_MAP = {
        'A': 'U',
        'B': 'F',
        'C': 'R',
        'D': 'B',
        'E': 'L',
        'F': 'D',
    }

    state = np.array(initial_state)
    solution_state = np.array(solution_state)
    print(state)
    print(solution_state)

    if solution_state[0] == "N0":
        m = {}
        faces = "ABCDEF"
        for j in range(6):
            for i in range(n ** 2):
                m["N" + str(j * n ** 2 + i)] = faces[j]

        state = np.array([m[s] for s in state])

    print("INITIAL", state)

    if n % 2 == 0:
        center_orienting_seq = []
    else:
        state, center_orienting_seq = orient_centers(state, moves, n)

    print("ORIENTED", state)
    print("ORIENT_CENTERS", center_orienting_seq)

    n2 = n ** 2
    NORMAL_SOLUTION = "A" * n2 + "B" * n2 + "C" * n2 + "D" * n2 + "E" * n2 + "F" * n2

    solution_faces = state_to_faces(solution_state, n)
    normal_solution_faces = state_to_faces(NORMAL_SOLUTION, n)

    print("SOLUTION FACES")
    print_faces(solution_faces, n)

    print("NORMAL SOLUTION FACES")
    print_faces(normal_solution_faces, n)

    move_map = get_move_map(n)
    print(move_map)

    print(state)
    faces = state_to_faces(state, n)
    print("INITIAL FACES")
    print_faces(faces, n)

    # Take the pseudo 3x3 formed by the corners, center edges, and the exact centers
    face_indices = [
        0, n // 2, n - 1,
        n // 2 * n, n // 2 * (n + 1), n // 2 * n + n - 1,
        (n - 1) * n, (n - 1) * n + n // 2, n * n - 1,
    ]
    indices = []
    for base_idx in [0, n*n, 2*n*n, 3*n*n, 4*n*n, 5*n*n]:
        for idx in face_indices:
            indices.append(base_idx + idx)

    state = state[indices]
    n = 3

    state = "".join(STICKER_MAP[c] for c in state)
    faces = state_to_faces(state, n)

    print("INITIAL FACES")
    print_faces(faces, n)

    print(faces)
    cubestring = make_cubestring(faces)
    print(cubestring)

    SOLVER_PATH = "/Users/Win33/Documents/Programming/rubiks-cube-NxNxN-solver/rubiks-cube-solver.py"
    cmd = [SOLVER_PATH, "--state", cubestring]

    out = subprocess.check_output(cmd)

    out = out.decode("utf-8").strip()
    out = out.split("\n")

    # Search for the solution line
    for line in out:
        if line.startswith("Solution: "):
            sol = line.split(":")[1].strip()
            break

    # sol = "D' B  L' U' D2 F' L  U  L  U2 F' L' D2 L  F2 R' L' D2 F2"

    n = int(puzzle_type.split("/")[-1])
    print(sol)

    # Map it back to our move set
    mapped_sol = []
    for move in sol.split():
        mapped_sol.append(move_map[move])

    solution = center_orienting_seq + ".".join(mapped_sol).split(".")

    mapped_sol = (".".join(center_orienting_seq) + "." if len(center_orienting_seq) > 0 else "") + ".".join(mapped_sol)
    print(mapped_sol)

    print(f"Validating")
    state = np.array(puzzle["initial_state"].split(";"))
    for move_name in solution:
        state = state[moves[move_name]]

    num_difference = evaluate_difference(state[indices], solution_state[indices])

    assert num_difference == 0

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()

pseudo(args.id)
