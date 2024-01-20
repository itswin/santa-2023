#!/usr/bin/env python3

import argparse
import pandas as pd
from util import *
import re

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
# parser.add_argument("moves", type=str)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
print(puzzle)

puzzle_type = puzzle["puzzle_type"]
n = int(puzzle_type.split("/")[-1])
moves = get_moves(puzzle["puzzle_type"])
# print(f"Number of moves: {len(moves)}")

# initial_state = puzzle["initial_state"].split(";")
# solution_state = puzzle["solution_state"].split(";")

with open(f"data/solutions/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

# with open(args.moves, "r") as fp:
#     moves = fp.read().split()

santa_to_sse = {}

sse_to_santa = {}

flip_move = {
    "U": "D",
    "D": "U",
    "L": "R",
    "R": "L",
    "F": "B",
    "B": "F"
}

for move in moves:
    regex = re.compile(r"(-?)([a-z])(\d+)")
    match = regex.match(move)
    
    inverse = match.group(1) == "-"
    move_type = match.group(2).upper()
    layer = int(match.group(3))
    add_invert = "'" if inverse else ""

    if layer >= n // 2:
        move_type = flip_move[move_type]
        layer = n - layer - 1
        add_invert = "" if inverse else "'"

    if layer == 0:
        # r0 to R
        # r5 to L
        santa_to_sse[move] = move_type + add_invert
    elif layer == 1:
        # r1 to NR
        santa_to_sse[move] = "N" + move_type + add_invert
    else:
        santa_to_sse[move] = "N" + move_type + str(layer) + add_invert

print(santa_to_sse)

sse_to_santa = {}

base_moves = {
    "F": "f",
    "R": "r",
    "D": "d",
    "U": "d",
    "B": "f",
    "L": "r",
}

# Normal moves
for move in "FRD":
    sse_to_santa[move] = f"{base_moves[move]}0"
    sse_to_santa[move + "'"] = f"-{base_moves[move]}0"
    sse_to_santa[move + "2"] = f"{base_moves[move]}0.{base_moves[move]}0"
for move in "ULB":
    sse_to_santa[move] = f"-{base_moves[move]}{n - 1}"
    sse_to_santa[move + "'"] = f"{base_moves[move]}{n - 1}"
    sse_to_santa[move + "2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 1}"

# Mid layer twists
for move in "FRD":
    sse_to_santa["M" + move] = f"{base_moves[move]}1"
    sse_to_santa["M" + move + "'"] = f"-{base_moves[move]}1"
    sse_to_santa["M" + move + "2"] = f"{base_moves[move]}1.{base_moves[move]}1"
for move in "ULB":
    sse_to_santa["M" + move] = f"-{base_moves[move]}{n - 2}"
    sse_to_santa["M" + move + "'"] = f"{base_moves[move]}{n - 2}"
    sse_to_santa["M" + move + "2"] = f"{base_moves[move]}{n - 2}.{base_moves[move]}{n - 2}"
for move in "FRD":
    sse_to_santa["N" + move] = f"{base_moves[move]}1"
    sse_to_santa["N" + move + "'"] = f"-{base_moves[move]}1"
    sse_to_santa["N" + move + "2"] = f"{base_moves[move]}1.{base_moves[move]}1"
for move in "ULB":
    sse_to_santa["N" + move] = f"-{base_moves[move]}{n - 2}"
    sse_to_santa["N" + move + "'"] = f"{base_moves[move]}{n - 2}"
    sse_to_santa["N" + move + "2"] = f"{base_moves[move]}{n - 2}.{base_moves[move]}{n - 2}"

# Slice twists
for move in "FRD":
    slice = "S" + move
    sse_to_santa[slice] = f"{sse_to_santa[move]}.{invert(sse_to_santa[flip_move[move]])}"
    sse_to_santa[slice + "'"] = f"{sse_to_santa[move + "'"]}.{invert(sse_to_santa[flip_move[move] + "'"])}"
    sse_to_santa[slice + "2"] = f"{sse_to_santa[slice]}.{sse_to_santa[slice]}"
for move in "ULB":
    slice = "S" + move
    sse_to_santa[slice] = f"{sse_to_santa[move]}.{invert(sse_to_santa[flip_move[move]])}"
    sse_to_santa[slice + "'"] = f"{sse_to_santa[move + "'"]}.{invert(sse_to_santa[flip_move[move] + "'"])}"
    sse_to_santa[slice + "2"] = f"{sse_to_santa[slice]}.{sse_to_santa[slice]}"

# Wide layer twists
for move in "FRD":
    sse_to_santa["W" + move] = f"{base_moves[move]}2.{base_moves[move]}1"
    sse_to_santa["W" + move + "'"] = f"-{base_moves[move]}2.-{base_moves[move]}1"
    sse_to_santa["W" + move + "2"] = f"{base_moves[move]}2.{base_moves[move]}2"
for move in "ULB":
    sse_to_santa["W" + move] = invert(sse_to_santa["W" + flip_move[move]])
    sse_to_santa["W" + move + "'"] = sse_to_santa["W" + flip_move[move]]
    sse_to_santa["W" + move + "2"] = sse_to_santa["W" + flip_move[move] + "2"]

# Tier twists
for move in "FRD":
    sse_to_santa["T" + move] = f"{base_moves[move]}0.{base_moves[move]}1"
    sse_to_santa["T" + move + "'"] = f"-{base_moves[move]}0.-{base_moves[move]}1"
    sse_to_santa["T" + move + "2"] = f"{base_moves[move]}0.{base_moves[move]}1.{base_moves[move]}0.{base_moves[move]}1"
for move in "ULB":
    sse_to_santa["T" + move] = f"-{base_moves[move]}{n - 1}.-{base_moves[move]}{n - 2}"
    sse_to_santa["T" + move + "'"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}"
    sse_to_santa["T" + move + "2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}.{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}"

print(sse_to_santa)
# with open(args.moves, "r") as fp:
#     solution = fp.read().split()

scramble = invert(current_solution)
sse_solution = []

for move in scramble:
    sse_solution.append(santa_to_sse[move])

print(" ".join(sse_solution))

# sse_solution = "D R2 L' SF' MU2 SR2 SF2 MD2 SR2 F2 B U2 B' ML D' MF' U' MF D L2 MF' U MF ML' U' L2 U MR' D' MB U' MB' D L2 MB U MB' MR U' L2 U R' L' F2 L R B L' R' F2 L R B'".split()
sse_solution = """

R' U B2 L2 MB' MD' B MD MB MD' B' MR' B2 MF MU F2 MF' MU ML' MU' ML F2 MU' B2 WR MU L2 MU' ML MU L2 MU' R2 MR MU' R MU MR' MU MB2 MU2 MB2 R' MU R2 U' MU ML' U2 MB' U2 ML U2 MU' F2 MU MB MU' F2 U' MD F U F' MD' F MU L MU' L' SU' L MU L' MU' D' F' D' F' L F MR B L' MB' L B' F' L' F MB MR' D ML SF' L SF ML' SF' L' SF U2 L MD' L' U2 L MD L' TU F' U R F MU' F' MU R' TU' F' U' F2 U B F2 U' F2 U B' F2 U' R' U2 R F' D2 F R' U2 R F' D2 F'
""".split()
# move_map = get_move_map(n)
santa_solution = []
for move in sse_solution:
    santa_solution.append(sse_to_santa[move])

santa_solution = ".".join(santa_solution).split(".")
# santa_solution = invert(santa_solution)

print(".".join(santa_solution))

# move_map = get_inverse_move_map(n, False)
# print(move_map)
# for move in santa_solution:
#     print(move_map[move], end=" ")

# print()

# print(".".join(current_solution))
