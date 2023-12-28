#!/usr/bin/env python3
import argparse
import pandas as pd
import twophase.solver as sv
import itertools
import numpy as np
from util import *

# seq 30 129 | xargs -P 4 -I {} python3 scripts/two_phase.py --id {}

def state_to_faces(state_string):
    return {
        "U": state_string[0:9],
        "F": state_string[9:18],
        "R": state_string[18:27],
        "B": state_string[27:36],
        "L": state_string[36:45],
        "D": state_string[45:54],
    }

def centers_aligned(state):
    return state[4] == "A" and state[13] == "B" and state[22] == "C" and state[31] == "D" and state[40] == "E" and state[49] == "F"

def extend_move_seq(seq, moves):
    for move in moves:
        yield seq + [move]

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, required=True)
args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]

print(puzzle)
moves = get_moves(puzzle["puzzle_type"])
print(f"Number of moves: {len(moves)}")

STICKER_MAP = {
    'A': 'U',
    'B': 'F',
    'C': 'R',
    'D': 'B',
    'E': 'L',
    'F': 'D',
}

LIST = "URFDLB"

MOVE_MAP = {
    'R1': 'r0',
    'R2': 'r0.r0',
    'R3': '-r0',
    'L1': '-r2',
    'L2': 'r2.r2',
    'L3': 'r2',
    'D1': 'd0',
    'D2': 'd0.d0',
    'D3': '-d0',
    'U1': '-d2',
    'U2': 'd2.d2',
    'U3': 'd2',
    'F1': 'f0',
    'F2': 'f0.f0',
    'F3': '-f0',
    'B1': '-f2',
    'B2': 'f2.f2',
    'B3': 'f2',
}

state = np.array(puzzle["initial_state"].split(";"))
print(state)

# Find the smallest number of moves to place the centers in the correct spot
center_moves = {}
for move in moves.keys():
    if move.endswith("1"):
        center_moves[move] = moves[move]
 
# Try longer sequences of moves if the centers are not aligned
seqs = [[]]
new_seq = []
while not centers_aligned(state):
    new_seqs = []
    for seq in seqs:
        for new_seq in extend_move_seq(seq, center_moves.keys()):
            new_state = state.copy()
            for move in new_seq:
                new_state = new_state[center_moves[move]]
            if centers_aligned(new_state):
                print("Found solution", new_seq)
                state = new_state
                break
            else:
                new_seqs.append(new_seq)
        if centers_aligned(state):
            break
    seqs = new_seqs

sol_state = puzzle["solution_state"].split(";")
print(state)
print(sol_state)
# UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
state_string = "".join(STICKER_MAP[c] for c in state)
faces = state_to_faces(state_string)
print(faces)
cubestring = faces["U"] + faces["R"] + faces["F"] + faces["D"] + faces["L"] + faces["B"]
print(cubestring)
solve = sv.solve(cubestring,19,2)

print(solve)
sol = (".".join(new_seq) + "." if len(new_seq) > 0 else "") + ".".join([MOVE_MAP[m] for m in solve.split(" ")[:-1]])
print(sol)
print("Done")

# Write the solution to a file
with open(f"data/solutions/{args.id}.txt", "w") as fp:
    fp.write(sol)
