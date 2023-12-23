#!/usr/bin/env python3
import argparse
import pandas as pd
from get_moves import get_moves
import itertools
import numpy as np
import subprocess

# seq 150 199 | xargs -P 4 -I {} python3 scripts/NxN.py {}

def state_to_faces(state_string, n=4):
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

#  D' L2 Bw2 Rw' D Fw' D' Rw2 U Bw2 D B 
# L Dw2 Rw' L2 U' F D2 Dw2 Rw2 F' L2 D' 
# Bw2 Lw2 D2 F2 R2 D' Lw2 U L D B2 L' D 
# L B' D2 F' R2 B2 D R2 U2 R2 D2 B2 L2 D'
move_map_4x4 = {
    "F": "f0",
    "F'": "-f0",
    "F2": "f0.f0",
    "Fw": "f0.f1",
    "Fw'": "-f0.-f1",
    "Fw2": "f0.f0.f1.f1",
    "B'": "f3",
    "B": "-f3",
    "B2": "f3.f3",
    "Bw'": "f3.f2",
    "Bw": "-f3.-f2",
    "Bw2": "f3.f3.f2.f2",
    "R": "r0",
    "R'": "-r0",
    "R2": "r0.r0",
    "Rw": "r0.r1",
    "Rw'": "-r0.-r1",
    "Rw2": "r0.r0.r1.r1",
    "L'": "r3",
    "L": "-r3",
    "L2": "r3.r3",
    "Lw'": "r3.r2",
    "Lw": "-r3.-r2",
    "Lw2": "r3.r3.r2.r2",
    "D": "d0",
    "D'": "-d0",
    "D2": "d0.d0",
    "Dw": "d0.d1",
    "Dw'": "-d0.-d1",
    "Dw2": "d0.d0.d1.d1",
    "U'": "d3",
    "U": "-d3",
    "U2": "d3.d3",
    "Uw'": "d3.d2",
    "Uw": "-d3.-d2",
    "Uw2": "d3.d3.d2.d2",
}

state_string = "".join(STICKER_MAP[c] for c in initial_state)
faces = state_to_faces(state_string, n)

print("INITIAL", state_string)

for face in faces:
    print(face)
    for row in chunks(faces[face], n):
        print("\t", " ".join(row))

cubestring = make_cubestring(faces)
print(cubestring)

# sol = "Dw2 Rw' D R2 Uw R' Fw' F' B' Rw' Bw2 Rw' Dw2 B' L' U Rw' Rw2 D B' Uw2 D B Dw2 B2 L2 D2 R2 B2 D' F2 Fw2 U' B2 Lw2 U2 D R B' U' B' U D2 R2 F R U' R2 U L2 B2 D' B2 D2"

# sol = sol.split(" ")
# print(sol)

# mapped_sol = []
# for move in sol:
#     mapped_sol.append(_4x4_move_map[move])

# print(".".join(mapped_sol))

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
    mapped_sol.append(move_map_4x4[move])

mapped_sol = ".".join(mapped_sol)
print(mapped_sol)

# Write it to the solution file
with open(f"data/solutions/{args.id}.txt", "w") as f:
    f.write(mapped_sol)
