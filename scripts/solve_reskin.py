#!/usr/bin/env python3
import argparse
import pandas as pd
from get_moves import get_moves
import itertools
import numpy as np
import subprocess
from util import *

# seq 210 234 | xargs -P 4 -I {} python3 scripts/solve_reskin.py {}

def reskin(state, edges, edge_map, odd_centers, odd_center_reskin_map):
    new_state = state.copy()

    for edge in edges:
        print(edge, state[edge[0]] + state[edge[1]])
        new_state[edge[0]] = edge_map[state[edge[0]] + state[edge[1]]][0]
        new_state[edge[1]] = edge_map[state[edge[0]] + state[edge[1]]][1]
        print(new_state[edge[0]] + new_state[edge[1]])
    
    for odd_center in odd_centers:
        new_state[odd_center] = odd_center_reskin_map[state[odd_center]]

    return new_state

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
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

edge_map_3x3 = {
    "BC": "AB",
    "BD": "AC",
    "BE": "AD",
    "BF": "AE",
    "CD": "BC",
    "DE": "CD",
    "EF": "DE",
    "FC": "EB",
    "AC": "FB",
    "AD": "FC",
    "AE": "FD",
    "AF": "FE",
}

edge_indexes = [
    (7, 10),  # UF
    (5, 19),  # UR
    (1, 28),  # UB
    (3, 37),  # UL
    (14, 21), # FR
    (23, 30), # RB
    (32, 39), # BL
    (41, 12), # LF
    (46, 16), # DF
    (50, 25), # DR
    (52, 34), # DB
    (48, 43), # DL
]

# Insert the reverse of each edge
for edge in list(edge_map_3x3.keys()):
    edge_map_3x3[edge[::-1]] = edge_map_3x3[edge][::-1]

state = np.array(initial_state)
solution_state = np.array(solution_state)
print(state)

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

edges = get_edges(n)
odd_centers = get_diff_odd_centers(n)

print(f"Edges {len(edges)}", edges)
print(odd_centers)

edge_map = make_edge_reskin_map(edges, solution_state, NORMAL_SOLUTION)
print(edge_map)

odd_center_map = make_odd_center_reskin_map(odd_centers, solution_state, NORMAL_SOLUTION)
print(odd_center_map)

# Make sure no edge indexes are in the odd_centers
assert len(set(odd_centers) & set(itertools.chain(*edges))) == 0

print(state)
faces = state_to_faces(state, n)
print("INITIAL FACES")
print_faces(faces, n)

state = reskin(state, edges, edge_map, odd_centers, odd_center_map)
print("RESKINNED", state)
print(type(state))

state = "".join(STICKER_MAP[c] for c in state)
faces = state_to_faces(state, n)

print("INITIAL FACES")
print_faces(faces, n)

print(faces)
cubestring = make_cubestring(faces)
print(cubestring)

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
