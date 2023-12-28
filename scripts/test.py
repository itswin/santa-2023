#!/usr/bin/env python3
import argparse
import pandas as pd
import os
import numpy as np
from util import *

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("--get_c_str", action="store_true")
parser.add_argument("--invert", action="store_true")
args = parser.parse_args()

sol_file_name = f"data/solutions/{args.id}.txt"

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]

with open(sol_file_name, 'r') as sol_file:
    solution_state = sol_file.read()
    solution = solution_state.split('.')

if args.invert:
    s = " ".join([invert(move) for move in solution])
elif args.get_c_str:
    s = "{"
    for move in solution:
        s += f"\"{move}\","
    s = s[:-1] + "}"
else:
    raise Exception("No flag specified")

# Write to file
with open(f"data/tmp_{args.id}.txt", 'w') as tmp_file:
    tmp_file.write(s)
