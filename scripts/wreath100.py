#! /usr/bin/env python3

import argparse
from collections import Counter
import pandas as pd
import itertools
import numpy as np
import subprocess
from util import *
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=337)
args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
puzzle_type = puzzle["puzzle_type"].replace("/", "_")

solution_state = np.array(puzzle["solution_state"].split(";"))

moves = get_moves(puzzle["puzzle_type"])

print(moves)
l_move = moves["l"]
r_move = moves["r"]

print(l_move)

identity = np.arange(len(solution_state))

l_moved = l_move != identity
r_moved = r_move != identity

print("L moved", l_moved)
print("R moved", r_moved)

both_moved = l_moved & r_moved

print("Both moved", both_moved)
# Get the True indices of both_moved
both_moved_indices = np.where(both_moved)[0]
print("Both moved indices", both_moved_indices)
