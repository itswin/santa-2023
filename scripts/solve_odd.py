#!/usr/bin/env python3
import argparse
import pandas as pd
from get_moves import get_moves
import itertools
import numpy as np
import subprocess

# seq 210 234 | xargs -P 4 -I {} python3 scripts/solve_odd.py {}

def state_to_faces(state_string, n):
    n2 = n ** 2
    return {
        "U": state_string[0:n2],
        "F": state_string[n2:2 * n2],
        "R": state_string[2 * n2:3 * n2],
        "B": state_string[3 * n2:4 * n2],
        "L": state_string[4 * n2:5 * n2],
        "D": state_string[5 * n2:6 * n2],
    }

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def make_cubestring(faces):
    return "".join(faces["U"] + faces["R"] + faces["F"] + faces["D"] + faces["L"] + faces["B"])

def get_center_slice_moves(moves, n):
    half_n = str(n // 2)
    center_slice_moves = {}
    for move in moves.keys():
        if move.endswith(half_n):
            center_slice_moves[move] = moves[move]
    return center_slice_moves

def centers_aligned(state, n):
    n2 = n ** 2
    return \
        state[n2 // 2] == "A" and \
        state[n2 + n2 // 2] == "B" and \
        state[2 * n2  + n2 // 2] == "C" and \
        state[3 * n2 + n2 // 2] == "D" and \
        state[4 * n2 + n2 // 2] == "E" and \
        state[5 * n2 + n2 // 2] == "F"

def extend_move_seq(seq, moves):
    for move in moves:
        yield seq + [move]

def orient_centers(state, moves, n):
    center_slice_moves = get_center_slice_moves(moves, n)

    # Try longer sequences of moves if the centers are not aligned
    seqs = [[]]
    new_seq = []
    while not centers_aligned(state, n):
        new_seqs = []
        for seq in seqs:
            for new_seq in extend_move_seq(seq, center_slice_moves.keys()):
                new_state = state.copy()
                for move in new_seq:
                    new_state = new_state[center_slice_moves[move]]
                if centers_aligned(new_state, n):
                    print("Found solution", new_seq)
                    state = new_state
                    break
                else:
                    new_seqs.append(new_seq)
            if centers_aligned(state, n):
                break
        seqs = new_seqs
    return state, new_seq

def get_move_map(n):
    base_moves = {
        "F": "f",
        "R": "r",
        "D": "d",
        "U": "d",
        "B": "f",
        "L": "r",
    }
    move_map = {}
    # "F": "f0",
    # "F'": "-f0",
    # "F2": "f0.f0",
    # "Fw": "f0.f1",
    # "Fw'": "-f0.-f1",
    # "Fw2": "f0.f0.f1.f1",
    for move in "DFR":
        # Number of layers
        for i in range(1, n // 2 + 1):
            if i == 1:
                move_map[f"{move}"] = f"{base_moves[move]}0"
                move_map[f"{move}'"] = f"-{base_moves[move]}0"
                move_map[f"{move}2"] = f"{base_moves[move]}0.{base_moves[move]}0"
            elif i == 2:
                move_map[f"{move}w"] = f"{base_moves[move]}0.{base_moves[move]}1"
                move_map[f"{move}w'"] = f"-{base_moves[move]}0.-{base_moves[move]}1"
                move_map[f"{move}w2"] = f"{base_moves[move]}0.{base_moves[move]}0.{base_moves[move]}1.{base_moves[move]}1"
            else:
                raise ValueError(f"Invalid number of layers: {i}")
    for move in "BUL":
        # Number of layers
        for i in range(1, n // 2 + 1):
            if i == 1:
                move_map[f"{move}"] = f"-{base_moves[move]}{n - 1}"
                move_map[f"{move}'"] = f"{base_moves[move]}{n - 1}"
                move_map[f"{move}2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 1}"
            elif i == 2:
                move_map[f"{move}w"] = f"-{base_moves[move]}{n - 1}.-{base_moves[move]}{n - 2}"
                move_map[f"{move}w'"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}"
                move_map[f"{move}w2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}.{base_moves[move]}{n - 2}"
            else:
                raise ValueError(f"Invalid number of layers: {i}")
    return move_map

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
print(puzzle)

puzzle_type = puzzle["puzzle_type"]
n = int(puzzle_type[-1])
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
print(state)

print("INITIAL", state)

state, center_orienting_seq = orient_centers(state, moves, n)

print("ORIENTED", state)
print("ORIENT_CENTERS", center_orienting_seq)

state = "".join(STICKER_MAP[c] for c in state)
faces = state_to_faces(state, n)

for face in faces:
    print(face)
    for row in chunks(faces[face], n):
        print("\t", " ".join(row))

cubestring = make_cubestring(faces)
print(cubestring)

move_map = get_move_map(n)
print(move_map)

SOLVER_PATH = "/Users/Win33/Documents/Programming/rubiks-cube-NxNxN-solver/rubiks-cube-solver.py"
cmd = [SOLVER_PATH, "--state", cubestring]
# cmd = ["cat", "scripts/split.py"]

out = subprocess.check_output(cmd)

out = out.decode("utf-8").strip()
out = out.split("\n")

# Search for the solution line
for line in out:
    if line.startswith("Solution: "):
        sol = line.split(":")[1].strip()
        break

print(sol)

# Map it back to our move set
mapped_sol = []
for move in sol.split(" "):
    mapped_sol.append(move_map[move])

mapped_sol = (".".join(center_orienting_seq) + "." if len(center_orienting_seq) > 0 else "") + ".".join(mapped_sol)
print(mapped_sol)

# Write it to the solution file
with open(f"data/solutions/{args.id}.txt", "w") as f:
    f.write(mapped_sol)
