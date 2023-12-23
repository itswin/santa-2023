#!/usr/bin/env python3
import argparse
import pandas as pd
from get_moves import get_moves
import itertools
import numpy as np


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

parser = argparse.ArgumentParser()
parser.add_argument("problem_id", type=int)
args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.problem_id]
print(puzzle)

moves = get_moves(puzzle["puzzle_type"])
print(f"Number of moves: {len(moves)}")

initial_state = puzzle["initial_state"].split(";")
solution_state = puzzle["solution_state"].split(";")

faces = state_to_faces(initial_state)

print("INITIAL", initial_state)

for face in faces:
    print(face)
    for row in chunks(faces[face], 4):
        print("\t", " ".join(row))
