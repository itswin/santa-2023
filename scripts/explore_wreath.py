#!/usr/bin/env python3
import pandas as pd
import json
import numpy as np
import argparse
import heapq
import time
from typing import Dict, Tuple, List
import sys
import random
from util import *

# id = 337
# puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[id]
# moves = get_moves(puzzle['puzzle_type'])

# print(puzzle)

# initial_state = np.array(puzzle["initial_state"].split(";"))
# solution_state = np.array(puzzle["solution_state"].split(";"))

# difference = evaluate_difference(initial_state, solution_state)
# print(f"Initial difference: {difference}")

# moves = get_moves(puzzle['puzzle_type'])
# moves = expand_moves(moves)
# print(moves)

# alg = "l.-r.l.-r.r.r.-l.-r.-l"

bottom_comm = "l." * 25 + "-r." * 26 + "-l." * 25 + ("r." * 26)[:-1]
# Swaps 125 and 175; 0 and 25
top_comm = "-l." * 25 + "r." * 26 + "l." * 25 + ("-r." * 26)[:-1]

def flip(n):
    return "l.-r." * n + "r." * n + "-l.-r." * (n - 1) + "-l"

# Flips Top left of right ring and bottom right
def flip2(n):
    return "-r.l." * n + "-l." * n + "r.l." * (n - 1) + "r"

# Flips Top left of left ring and bottom right
def flip3(n):
    return "-l.r." * n + "-r." * n + "l.r." * (n - 1) + "l"


# Flips (Top right of left ring)
alg = "l.-r.l.-r.l.-r.l.-r.l.-r.r.r.r.r.r.-l.-r.-l.-r.-l.-r.-l.-r.-l"
# assert alg == flip(5)
# alg = flip(3)
alg = flip3(9)
# alg = "l.r.l.r.l.r.l.r.-r.-r.-r.-r.-l.r.-l.r.-l.r.-l"

# Flips 194-197 (Top left of right ring)
# alg = "-r.l.-r.l.-r.l.-r.l.-l.-l.-l.-l.r.l.r.l.r.l.r"

# alg = invert(alg)
# alg = list(reversed(alg.split(".")))

# Rotates L 4 places CC and R 4 places CC
# alg = "l.r.l.r.l.r.l.r.-l.-l.-l.-l.-r.l.-r.l.-r.l.-r.l.r.r.r.r"
# alg = invert(alg)
# alg = "l.r.-l.-r"
# alg = "l.r.l.r.-l.-r.-l"

# Swaps n with Top left first 
# alg = "l.r.l.r.l.r.-l.-r.-l.-r.-l"

# Swaps n with Top right first
# alg = "l.-r.l.-r.l.-r.-l.r.-l.r.-l"

print(alg)

# indexed_solution = []
# for i in range(len(solution_state)):
#     indexed_solution.append(f"N{i}={solution_state[i]}")

# indexed_solution = np.array(indexed_solution)

# # print(indexed_solution)

# alg = alg.split(".")

# state = indexed_solution.copy()
# for move in alg:
#     state = state[moves[move]]

# print(state)
# print_wrong_stickers(state, indexed_solution)
