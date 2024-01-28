#!/usr/bin/env python3

import argparse
import pandas as pd
from util import *
import re

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("--partial_sol", type=str, default=None)
parser.add_argument("--use_minus", action="store_true", default=False)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
print(puzzle)

puzzle_type = puzzle["puzzle_type"]
n = int(puzzle_type.split("/")[-1])
moves = get_moves(puzzle["puzzle_type"])
# print(f"Number of moves: {len(moves)}")

initial_state = np.array(puzzle["initial_state"].split(";"))
solution_state = np.array(puzzle["solution_state"].split(";"))

with open(f"data/solutions/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

santa_to_sse = get_santa_to_sse_move_map(n)
# print("Santa to SSE")
# print(santa_to_sse)
# for k, v in santa_to_sse.items():
#     print(k, "\t", v)

sse_to_santa = get_sse_to_santa_move_map(n)
# print("SSE to Santa")
# print(sse_to_santa)
# for k, v in sse_to_santa.items():
#     print(k, "\t", v)

scramble = invert(current_solution)
# scramble = []

# state = solution_state
# for move_name in scramble:
#     state = state[moves[move_name]]

# assert np.all(state == initial_state)

if args.partial_sol:
    with open(args.partial_sol, "r") as fp:
        sol = fp.read()
        delimiter = "." if "." in sol else " "
        partial_sol = sol.split(delimiter)

    scramble.extend(partial_sol)
    print(f"Applying partial solution of length {len(partial_sol)}: {partial_sol}")
else:
    partial_sol = None

# if n % 2 == 0:
#     center_orienting_seq = []
# else:
#     initial_state, center_orienting_seq = orient_centers(initial_state, moves, n, solution_state)
#     print("Center orienting seq")
#     print(center_orienting_seq)
#     scramble += center_orienting_seq

move_map = get_inverse_move_map(n, args.use_minus)
# print(move_map)
cube_scramble = " ".join(scramble)
# cube_scramble = list(map(lambda x: move_map[x], cube_scramble.split()))
cube_scramble = " ".join(cube_scramble)
print("\nCube scramble")
print(cube_scramble)
print()

sse_scramble = []
for move in scramble:
    # print(move, "\t", santa_to_sse[move])
    sse_scramble.append(santa_to_sse[move])
print("\nSSE Scramble")
# print(" ".join(sse_scramble))
print(len(sse_scramble))
print(sse_scramble[:10])
print()

with open(f"data/sse_scramble.txt", "w") as f:
    f.write(" ".join(sse_scramble))

sse_scramble = """
MR' MF' MR MF MR N6B' N7D N8U' N9F' L MR' N9D' MR D N9D L2 N9D' L2 MR' N9D MR N9D' D' L2 N9D L N9U L' MF R' MU' N9R' MU N9R R MF' L N9U' D' MF' N9U' B N9U' MF N9U B' D N9U MU F N9U' F' MU' F N9U F' N9D' B' N9D B N9D' N9B2 N9U N9D B' N9D' B N9D2 N9U' N9B2 R' N9R' N9F R N9D' R' N9F' N9D' N9R N9D R N9L F N9D N9U N9F U N9F' N9U' N9D' N9F U' N9F' F' N9L N9F' D N9D N9F' N9D' N9F D' N9F N9L2 U2 D N8L' U2 N9F U2 N8L U2 N9F' D' N8F N9R' N9D' N8F' D' N8F D N9D N9R D' N8F' D2 U' N8F' D' L' N9U N9F' L D N8F D' L' N9F N9U' L U2 N8B U' N9B' U N8B' U' N9B N9U U' N7F L' N9F' N9U L N7F' L' N9U' N9F L U N9U' N9L' L' N7D N7B L N7B' N9F N9R' N7B L' N7B' N9R N7D' L N9F' N9L N9D N7B' N9D' F N9D N7B N9D' F' N7U' B' N7U N9B N9U' N7U' B N7U B' N9U N9B' B N7L' N9B' R' N7L' B N7L N9B N7L' B' N9B' R N9B N7L' F N7L' N9B N7L F' N7L' N9B' N6B' N6L N9U L' N9U' L N6L' N6B' L' N9U L N9U' N6B2 N6R' N9B' N6R' B N6R N9B N6R' N9D' B' N6R2 N6U' B N9D B' N6U N9B N9U' D' N6F D N6F' N9U N9B' N6F D' N6F' D N9R U' L N9F L' N6D' N6B2 L N9F' L' N6B2 N6D U N9R' D' R N5D' N9L' N9F2 N5D R' N5D' N5U' R N9F2 R' N5U N9L N5D D N5B2 L' N9D N5L N9B L N9B' N5L' N5B' N9B L' N9B' N9D' L N5B' N9D N5R N9D' L' N9D N5R' N9D' L B' N9U' N5R N5D' N9U B N9U' B' N5D N5R' B N9U F N9R N5U N5R' N9R' F' N9R F N5R N5U' F' N9R' B N9R' N4B N9R B' N9R' N4B' N4F' N9R F' N9R' N4F N9R F N9D' N4L' B N4L B' N9D N9B' B N4L' B' N4L N9B N4F' R' N4D' N9L' N4D R U' N4D' N9L N4D N9L' U N9L N4F N4U2 L N9B L' N4U2 L N9B' N9F' L' D' N4B D L N9F L' D' N4B' D N3B' N9U' N3B' U N3B N9U N3B' U' N3F' N3B' L' N3B N9U N9L2 N3B' L N3B N9L2 N3F L' N9U' L N3B2 N3D' L N9F2 L' N3D N9F' N3L' N9F L N9F' N3L N9F' N9B L' N3U' L D N3B D' N9B' D N3B' D' L' N3U N9D N9F NL' D F' NL N9F' NL' N9F F D' N9F' NL NR' N9D' L N9D NR N9D' L' D' N9R NU N9R' D N9R NU' N9R' L' ND' L N9U N9B' L' ND L ND' N9B N9U' R' ND N9R ND' R ND N9R' U' N9L U NF' NL' U' N9L' U N9L NL NF N9L' MF L' N8U2 MF2 N8U MF2 L MF' L' N8U L N8L' R2 MF' N8U F N8U MF N8U' F' N8U' R2 N8L U N8B' MU N8B U N8B MU' N8B' N8F' MU N8F U2 N8F' MU' N8F R' D MF N8D' MF' N8D D' R B' N8U' N8L F' N9R' F N8L' N8U F' N9R F B N8D' N9L N8F L' N8F' N9L' N9B' N8F L N8F' N8D2 L' N9B L N8D' N9B U' N8B2 U R N9B R' U' N8B2 U N8U' R N9B' R' N8U N9B' N9F' D' N8R N8F D N9F D' N9F' N8F' N8R' N9F D N8R2 L N8D N8B' N8D' B N8R N8B N8R' B' L' N8B' N8D N8B N8D2 N8R U' N8R' N8D N8R U N8R F' N8F U N8F2 N8L N8F2 N8L' N8F' N8U N8F U' N8F' N8U' F U N8L D N8L N8B2 N8L' N8B2 D' N8L' U' F' N8D F N7D' F' N8D' F N8F D N8F' D' N7D N7B D N8F D' N8F' N7B' N8L' N7U' N8L U' N8L' N7U N8L U' N7D F' N7D' N8B N7D N8B' F U' N8B N7D' N8B' N7B N7R' U N8F U' N8F' N7R N7B' N8F U N8F' U2 B' N7R' F' N8U' N8R F N7R F' N8R' N8U F B L N8D' N6L N8D L' N8D' N6L' N8D N8U L N6U' L' N8U' L N6U2 N8B' D N8B N6U' N8L D' N6F' D N8L' N8B' D' N8B N6F L' N8U' D N6B2 D' R' N8B' R D N6B2 N6D' D' R' N8B R N6D N8U N6R' N6D D' N8B D N8B' N6D' N6R' N8B D' N8B' D N6R2 N8D' N5D' R N5D R' N8D N8R' R N5D' R' N5D N8R N5B D N5B N8D N5B' D' F N5D' F' N8D' F N8D N5D N5B N8D' N5B2 N8L N5D N5L' N8L' F' N8L F N5L N5D' F' N8L N5U' N8L U N8L' N5U N8L U' N8L N5U2 N8B' N8U' R N5B' R' N8U N8B N8U R N5B R' N8U' N5U2 B' N4U N4L' B' N8U B N4L N4U' N4L B' N8U' B N4L' B N8D' N4B' R N4B R' N8D N8R' R N4B' R' N4B N8R N4L' N8B N4L F' N4L' N8B' N4L N8B' N8R F N4R F' N8R' N4R' N8B N4R F N4R' N8F' D' N8D' L' N8D N4R N8D' L N8D D N8F D' N4R' D N3D N3F' N8L' L' B N8L N3F B' N3R B N8L' B' N3R' N3F' L N3F N3D' L' N3D N8L N3D' L N3R N8F N3R B N3R' N8F' N3R B' U' N8B2 U N3R2 N8B' N3U' N8B U' N8B' N3U N8B' U F N3L' F2 N8F' R N8D N8F F2 N3L F2 N8F' N8D' R' N8F F N8D' NR' N8D R N8D' NR N8D NR' N8U NR NL D2 NR' N8U' NR D2 R' N8U NL' R NU' R' N8U' R NU R' B' U' N8L U B NL ND B' U' N8L' U B ND' D N8R D' NL' D N8R' D' NB' N8D2 R2 NF R2 N8D2 R2 NF' R2 NB N7B' U' N7D D N7R D' MF' D N7R' N7D2 D' MF N7D R' N7D' MR N7D R N7D MR' U N7D' N7B D B N7R N7B' B' MU B N7B N7R' B' D' N7R N7B' MU' N7B N7R' MU F' N7L' N7U' F MU2 F' N7U F MU F' N7L F D N7F D' N9R N7R' N9D N7R D N7R' N9D' N7R N7F' D' N7D N9R' D' N9R N7D' N9R D N7F' D' N9R2 D N9U N7F U N7F' N9U' N7F U' N9D' B' N7D B N9D B' N7D' N9L N7B N9L' B N9L N7B' N9L' N7B' N9U F N7B U N7B' N9U' N7B U' F' N7U F N9U F' N9U' N7U' N7F N8L N7F' N8U R' N7U' R N8U' R' N7U N7F N8L' N7F' R N7D' N7L L F N7L' N8F N7L N8F' F' L' N8F N7L' N7D N8F' D L' N8F N7D' N8F' N7D L N7B N8U' N7B' D' N7B N8U N7B' N8R' U N7L' U' N8R U N7L L N7F' N8F' N8L' N7F L' N7F' L N8L N8F L' N7F U' N7D R' N7F2 R N7D2 N7B' R' N7F2 R N7B D N7F F D N7F' N7D N7F D' N7B N7D' F' N7D N7B' N7F' D' N7U R N7U' N7L' N7U R' N7U' N7L B N7U' N7F' U N7F N7U N7F' U' N7L' N7F2 N7L N7F' B' N7L' N7U L N7U' N7L N7U L' N7U' N6L' N7U N6L F' D' N6L' N7U' N6L N7U D F L N7U N6R N7U' L' N7U N6R' B N6L2 B' N7U' N6F N7U' B N7U N6F' N7U' N6L2 B' R2 F' N6U' N7B N6U N7B' F R2 N7B N6U' N7B' N6U L2 N6B' L N7D' N7F L' N6B L N7F' N7D L N5L D' N7L' N5L' N7U2 N5L D N5L' N7U2 D' N7L D N7R' N5U' N7R U N7R' N5U N7R U' D' N5F' N7B N5U' N7B' D N7B N5U N7B D' N5F D N7B' N5L N7B' R' N7B N5L' N7B' R N7R' U' N7R N7F N5D2 N7F' N7R' U N7R N7F N5D2 N7F' N7R' F N4U' F N7R' N7F' F' U' N4L U F N7F N7R F' U' N4L' U N4U F' N7R N4R' N7U' N4B' N7B' U N4R U' N7B U N4R' N4B N7U N4B' U' N4B N4R R N4L N7F' R' N4U2 N4B R N4B' N7F N7L' N4B R' N4B' N7L N4U2 N4L' N4D2 L N7F N4D' N7R N4D L' N4D' N7R' N4D' L N7F' L2 N7B N3R N7B' L N7B N3R' N3D N7B' U2 N7B N3D' N7B' U2 L N3D N3B L' F' N7D2 F L N3B' N3D' L' F' N7D2 F' N7U' N3F N7U F D' N7U' N3F' N7U N3F D N3F' N3L N7F N3L' F N3L N3F L' N7F' L N3F' L' N3L' N7F L N7F' N7B2 D' NL' D N7B2 D' NL D NL N7F' R NL' F NL N7F F' N7D F NL' F' N7D' N7F' R' N7F F' ND F N7U F' ND' F NR' U' NR N7U' NR' N7R' U NR2 U' N7R U N7D NR' D R' NR N7D' NR' N7D R D' N7D' NR U N7L U' NR' U N7L' U' N6L B N6D MF N6D' B' N6D' MF' N6D B' N6L MF' N6L B' N6L MF N6L' B N6L2 B N6L' L N6D' MR N6D L' N6D' MR' N6D N6R MR N6B R' N6B' MR' N6B R B' MU' N6B' MU B N6R' F N6D' L' N6F' MR' N6F' MR N6F2 L N6D F' D' N9R N6D' N9R' D N9R N6D N9R' F N6L F N9L' F' N6L' F N6B N9L F2 N9L' N6B' N9L N6R' F' N6R N9B' N6R' F N9B L N9B' N6R N9B N9R' U2 N9R N6F N9R' N6F' U2 L' N6F N9R N9F' N9D N6F D' N6F' N9D' N6L D N9F2 D' N6L' N6F D N9F' N6F2 D L' N8F N8R' L U N6B U' L' N8R N8F' L U N6B' U' D' N8F' N6R' B' N6R N8F N6R' B N6R' B R N6R N8D N6R' N8D' R' B' N8D N6R N8D' N6R N6L N8U B' N6U B N8U' N8L B' N6U' B N8L' N6L2 N8B2 N6L' F' N6L N8B2 N6L' N8R' F N6L2 F' N8R F N7D R' N6D R N7D' N7F R' N6D' R N7F' N7L N6F N7D' N6F' U' N6F N7D N6F U N7L' U' N6F2 U N7F N6R2 N6B' N7F' D' N7F D N6B N6R2 D' N7F' N7D R' N6B R N7D' N6R2 N7U' R' N7U N6R2 N6B' N7U' R N7U D L2 R' N6U' L N6B' N6U N6B N6U' L' N6U R L2 F2 N6L' F' N6U' N6L N6U N6L' F N6L F2 N6R B N6R' N6F' N6R B' N6R' N6B2 N6D' R N6D R' N6F N6R' R N6L' N6D' R' N6D N6L N6R N6B2 N6L' N5L' U' N6L N6F' U N5L2 U' N6F N5D' N6L' U N6L N5D N5L' N6R' B' N5L' B N6R B' N5L N6R' B N5R' B' N6R N6D B N5R2 B' N6D' B N5R' N6R2 N5F N6R F' N6R' N5F' N6R F N6R B' N6U' F' N6U U' N5B N6U' N5B' N6B' U N5B U' N6B U F N6U N5B' B N4B N4D' N6L D' L' N6L' N4D N6L N4D' L D N4D N6L' N4B' R L2 N4D R' N6D' N6B R N4D' R' N6B' N6D L2 N6U D N6L' D' N4R D N4R' N6L N6U2 N4R D' N4R' N6U N4F L' N4F' N6R N6D' N4F L N4F' L' N6D N6R' L N6L B N3L N3U N6B' N3U' N3L' B' N3L N3U N6B N3U' N3L' N6L' D' N6D' B' N6D N3F N6D' B N3F' D N3F N6D N3F' B N6R' N6D' B' N3U B N6D N6R B' N3U' N6U' N3L' N6U' R' N6U N3L N6U' R N6U' F N6B N3L' N6B' N3L F' R' N3L' N6B N3L N6B' R N3B' U' N3B N6U' N3B' U N3B B' NU N6B' NU' B NU N6B' N6L R U NB U' R' N6L' N6B' R U NB' U' R' N6B' NU' F2 N6R' NU N6R NU' F2 R NU N6R' NU' N6R2 R' ND L ND' N6R' ND L' ND' MR' MF N5L' MF' N5L D' F N5L MF N5L' F' MF' D MR N5B MU N5B' U' N5B MU' N5B' U D N5F' D' MR' D N5F D' R' N5D MR N5D' R MR L N5D' L N5D MR' N5D' L2 MU L N5D L' MU' D N9B D' N5D' N5L' D N9B' D' N9B N5L N5D N9B' R' N9U' N5L' R' N5U R N9U R' N5U' N9U' N5L N9U R2 N5R N5U D' N9L D F' N9L' F N5U' N5R' F' N9L F D' N9L' D N5R' U' N5R N9D' N5R' U N5R N9D U' N9L' N5D' N9L U N9L' N5D N9L F' N8F N8D N8R' F N5L F' N5L' N8R N8D' N8F' N5L F N5L' N5D D R N5D' N8B N5D N8B' R' D' N8B N5D' N5L' F2 R' N5L N8B' N5L' N8B R F2 N8B' N8U N5L U2 B' N5L' N8U' N5L N8U B U2 N5U R' N5U' N8D' N8F' N5U R N5U' R' N8F N8D R N8U' N5L F' N7R F N7R' N5L' N5B' N7R F' N7R' F N5B N7R' B' N5R B N5R' N7R N7B N5R B' N5R' B N7B U N7B N5U' N7B' U' N7B N5U N5R N7B L N7B' N5R' N7B L' N5F' N7F N5L N7D L' N5F' L N7D' N7F' L' N7F N5F N5L' N7F' L N5F N5U N7L' N5U' R' N5U N7L N5U' R N6B R N5F' F' N5R F N6R' F' N5R' F N5F R' N5F' N6R N5F N6L' N5L' D' N5L D N6L N6B' D' N5L' D N5L N5F' R' N5F N6L N5F' R N5F N6L' F2 N5R' N6U' N5R N6U F2 U N6U' N5R' N6U N5R N5B N5L' U' N6R2 U N5L U' N6R2 N6F U N5B' U' N6F' D' N5R R B' N5R N5F N5R' B N5R' N5F' R' D N5D2 N5F U N5F' N5D2 N5U' N5F U' N5F' N5B' D N5B N5U N5B' D' N5B U B N5B N5U' B' N5B2 N5D' N5B2 B N5U B' N5D N5B' U' N4F B N5U B2 N4L B2 N5U' B2 N4L' B N5D' N4F B L' N4F' N5D N4F N5D' L B' N5D N4F' N4U' B N5L' B' N5L N4U N4F' N5L' B N5L B' U' N4B N5D' N4B U N4B' N5D N4B N5F' U' N4B2 U D N5F N4U N5F' D' N5F N4U' N5L' N4D R' N4D' N5L N4D R N4D' L N3U' L' N5F' L N5F N3U N3R N5F' L' N5F N3D2 N5L F N5L' F' N3D2 N3R' F N5L F' N5L' N5U' N3L D' N3L' N5U N3L D N5D' R2 N3L' D N3L N5D N3L' D' N5D' R2 N5D N5B N3R N5B' D' N5B D N3R' N3B' D' N5B' D N3B N5D NF' NL2 D' N5B D NL2 D' N5B' NF N5D' NF' D N5L' NF R N5F R NF' R' N5F' R N5L NF R' N5B R' NF R N5B' R' NF' N5L' ND N5L' L' N5D' L ND' L' N5D ND N5L ND' N5L ND' L N5U' L' ND L F' N5U NB' N5U' F N5U NB B' N4F R N4F' MR' N4F MR R' B2 MR' N4F' MR B' MF2 R MF' N4D' MF N4D2 R' MF2 R N4D' R' MU' F' MU N4D N4B' MU' F MU F' N4B N4D' F D' N4U F' N4U' MF N4U F MF' D MF N4U2 MF' N4U N9U N4D2 N4R' N9U' R' N9U N4R N9U' N9F' R N4D2 N4B' R' N9F R N4B N9D' N9U F' N9D' N4L' N9D F N9U N9D' F' N4L F N9U2 N4B' N9D' B N9D B' N4B N4D2 B N9D' B' N4D2 SF N4U N4F' F' L' N9D' L F N4F N4U' F' L' N9D L B N9D' N4F U' N8L U N4F N4R U' N8L' U N4R' N4F2 D' N4B N8F D N4B' N4D' D' N8F' D N8F N4D N8F' N8D2 N4B D' N4B' N8D2 D B L N4F N8U' N4F' N8U L' B' N8U' N4F N8U N4F' N4B' N8R U' L N8D' L' N4D' L N4D N8D N8R' N4D' L' N8R U N8R' N4D N4B N7U N4B N7U F N7U' N4B' N7U F' N7U2 L' N7B' L N4D2 L' N7B N7U' L N4D' L' N7U N4D' N7L' N4D L N4D' N7D' F' N7D N4F N4D' N7D' F N7D F' N4D N4F' F N7L N4U R N7U' R' N4U' R N7U F N4D N7B N4D' N7B' F' R' N7B N4D N7B' N4D' N4R N4L B N6D2 B' N4L' N6F N4R' B N4R N6F' N6D2 N4R' U' N6L' N6B U N4F2 U' N6B2 U N4F' U' N6B N6L U N4F' N6U N4B' N6U' B' N6U N4B N6U' D' N6L' D N4R' D' N4R N6L N6U' N4R' D N4R N6U N6R' D' N6R N4D N6R' D N6R N4D' N5R N4U N5R' U N5R N4U' N5R' U' N5L' D N4R' D' N4R N5L N5U2 N4R' D N4R D' N5U2 N4F N5R N4F' L2 N4F N5R' N4F' F N4U' N5F N4R' F' N5D' F N4R N4U F' N4U' N5D N5F' N4U L2 D' N5R N4B' N5R' N4B D N4L' B' N4B' N5R N4B N5R' B N4L N4F N4D' N4L N4D N4L' F D' N4L N4D' N4L' N4D D F' N4F' U N4B N4D N4B' U' N4B N4D' B N4R2 N4B' D' N4R' N4D' N4R N4D D N4R2 B' N4U' N4F' R N4F N4U N4F' N4R N4U2 N4R' N4U R' N4U N4F R N3D R' N3D' N4L N4F' N3D R N3D' R' N4F N4L' N3U' N4B R' N4B' R N3U2 N3R R' N4B R N4B' N3R' L N3U' N4R N3U L' N3U' N4R' N3U' N4L F N3U2 F' N4L' F N3U' N4F' N3U' F' N3U N4F N3L' N4B' N3B U N3B' U' N4B N4D U N3B U' N3B' L' N4D' N3L N4D L N4D' R NU' R' F' N4L N4U' F R NU R' F' N4U N4L' F NR' N4R' B' N4R B NR NF' B' N4R' B N4R NF NL2 B' N4L NL N4B NL' B NL N4B' NL B' N4L' B U N4R ND' N4R' U' N4B' N4R U N4B ND N4B' U' N4R' N4B N3L N3F F N3L' U' MR U N3L F' N3F' N3L' F U' MR' U F L MU' N3B U' N3B MU N3B' U N3B' D' MR' N3D MR D L' MU' L N3D' L' MU F2 N3B' L' N3U' MR' N3U L N3U' MR N3U N3B MR' N3U B D MR N3U MR' N3U' D' B' N3U' MR N3D' N9D R' N3B' R N9D' N9B R' N3B R N9B' N3D L' N3U' L N9U L' N3U L N9U' N3F L N3F' N9L N3F L' N9L' F N9L N3F' N9L' N9R N3F' N9R' N3F F' R' N3F' N9R N3F N9R' N9U N3R N9U' R N9U N3R' R' N3L' D N3L N9U' N3L' D' N9U R N9U' N3L L N8R F2 N8R' N3D N8R N3D' F2 L' N3D N8R' N3D' B' N3D' N3L' B N8D' B' N3L N3D B N8D N3D' F N3D N8B' N3D' F' N3D N8B N3U F' N3U' N8F' N3U F N3U' N8B R N8B' N3L' N8B R' B N8L B' N3L B N8L' N3L' N8B' N3L B2 N8F N7R2 B N3D2 N3L' B' N7R' B N3L B' N7R' B N3D' L N3D' N7L' N7U N3L N7U' L' N7U N3L' L N3D L' N7U' N3D' N7L N7B' N7D N3R N3F' N7D' N3F D R' N3F' N7D N3F N3R' N7D' R N3R D' N3R' N7D' B' N7D N3D N3R' N3B' N7D' B N7D B' N3B N3R N3D' B N7B N3B N6D N3B N6D' N3B' B' U N3B N6D U' N6R U N3B' U' N6R' N6D' N3B' N3D' N6B' N3D B N3D' N6B F' N6U N3F N3U' N6U' F N6U F' N3U N3F' F N6U' N3D D' N6R N3D N6R' D N6R R' N3D' N6R' N3D R N3D' N6F' N3L' N6F R N6F' N3L N6U2 R' N3F' R N6U2 N6F R' N3F N5B' U N5D N3D N3L' N3F N5D' F N5D F' N3F' N3L N3D' F N5D' F' U' N5B N5U R N5F' N3L N5F R' U2 N5F' N3L' N5F N3L U2 N3L' N5U' N5D' B N5R N5U' B' N3U' B N3U N5U N5R' N3U' B' N3U N5D N3B2 D' N5L' D N3B2 N3R' D' N5L' D N3R N3B' D' N5L2 D N3B N3F N3R' N4U N3R U' N3R' N4U' U N4R U' N3R U N4R' N3F' N4D N3R N4D' R' N4D N3R' R N3U' R' N4D' R N3U N3F' N4L N3B' N4L' N4F N3L' B N3L N4F' N3L' N4L B' N3L B N3B N4L' B' L D' N4F' N3R' N4F N3R D L' N3R' N4F' N3R N4F N3F N3R' F' N3D F N3R F' N3D' F N3U N3F N3U' N3R F' N3R' F N3U N3F' F' N3B' N3R F N3R' N3B N3U' F N3B N3D' N3B' N3D F' U' N3D' N3B N3D N3B' U N3B D N3L N3F' N3L' F' N3D' N3L N3D N3L' N3F F D' N3B' N3F NU' N3F U N3F' NU N3F NL U' N3F2 U NL' N3B ND NL' N3B' U' N3B U NL ND' U' N3B' NR U' NR' N3D' NR U N3D' L N3D NR' N3D' L' N3D2 ND' L' VD N3B N3L' ND' L ND L' N3L N3B' N3D' L B' NU MF D' MR NU' MR' D MF' B R NU' R' MU R NU R' MU' NF2 R' ND MR ND' R' ND' MR' ND R2 B U NF MU' NF' U' NF' MU B' NF' L' B' MU' B MF ND' MF' ND' B' MU B ND2 L TU' L NU N9R NU' L' NU N9R' TF' NL' F U N9L2 U' F' NL TF U N9L N9F ND' B N9F' D N9F ND N9F' TD' B' ND N9L D N9D N9B R' N9D' U2 NL U2 N9D R N9B' N9D' U2 NL' U2 D' NB' N8R NB R' NB' N8R' NB N8D R NU2 R' N8D' N8B' NL N8B R N8B' NL' N8B NU L NU N8R' NU' L' NU N8R N8L' NL' D' NL D N8L N8U2 D' NL' D NL N8U U R' N8U NR' N8U' TR U' NR' N8U N7R' F N7R' F' NR ND F N7R F' N7R' ND' N7R' N7D' L NF N7D NL N7D' TL' U NL N7D NL' U' L NF' L' N7R' N7U NR NB2 N7U' R' N7U R NB2 TR' N7U' R U N7B N7L N7B U NF2 U' N7B' U NF' U' N7L' N7B' U NF' U2 NF N6U NL B N6L' B' NL' B N6L N6U' NF' N6U B' N6U' N6D NF N6D' F2 N6D NF' N6D' NR' N6F ND' B' ND N6F' N6U' ND' B ND NR B' N6U B F2 N6D NR' N6D L' N6D' NR N6D L D B NL2 B' D' N6F' N6R' D B NL2 B' D' N6R N6F N6D2 D' R NU' NF' R' N5B R NF NU R' N5B' D NB' NL N5L' U N5L U' NL' NB' U N5L' U' N5L NB2 D' N5R L ND N5R' ND' N5D L' ND L N5D' L' D N5R ND' N5R' R NB N5L NB' R' NB N5L' L' NB' N5L NB L NB' N5L' NB' R' N4U' N4L' B' NU B N4L N4U B' NU' B R NB N4U NL' N4U' L N4U NL L' NU2 L N4U' L' NU2 N4D NF D' NF' N4D' NF D NF' L N4D NB' N4D' NB L' NB' B N4D NB N4D' N4R' NF' N4R B' N4R' NF D' N4R ND N4R' D N4R ND' R2 NL D N3L' D' NL' NB D N3L D' NB' R2 F' N3D F NR2 F' N3D' N3L' F NR' F' N3L F TR' NF R N3F' R' ND' R N3F' R' ND NF' R N3F2 N3R' U N3R L' NU N3R' NU' L NU NB U' N3R U NB' TU' L' NU' NR' NF' NR' NF NR2 NU TL NB' NL D' NL' NB NU NB NL NB2 D NB2 NL' NB' NU' NL' L2 NB' NL' NB L NB' TL NF' L' NB L TF ND R ND' NL' ND R' ND NL' ND2 NL2 F' MF B' U' MF' R MF R' U B2 R MF' R' U' L' F' MR F L D' F' MR' F U MR D MR' R D' F' L' R' N9D R L F D R' L' F' N9D' F L B' N9B' R U' F U N9U N9R' U' F' U R' F R N9R N9U' R' F' N9B U' F' N9U' F U F' N9U2 F D R' F' N9U' F N9U R D' N9U' B R N9D R' N9D' B' D' N9D R N9D' R' D F R' N8U B' D' B N8U' L' N8D2 L B' D B L' N8D2 L R F2 R' N8D' R U D R' N8D R D' U' F' L' U' N8F' U N8F L F N8F' U' N8F U F D2 B' N8D' B D2 B' N8D B N8F' U' N8L' U F' R' U' F N8L F' U R F N8F D F L' N7B U' N7B' U L F' D2 L' F U' N7B U N7B' F' L D N7F' L F' L' N7F L F N7D R N7D' L' N7D L D' N7R D R' L2 D' N7R' D L N7D' N7B' R U' R' N7B R N7B' U N7B R' N7L B2 N7L' B' R2 B N7L B2 N7L2 B R2 B' N7L N6L B' R B N6L' B' D' R' N6U' R D R' N6U B N6B' U' F' U N6B U' F L B' N6L' B L' B' N6L B U N6B2 R N6F' L N6F N6U R' N6U' N6F' L' N6F N6U U R N6U' R' U' N6B2 R' F' R' N6F R F B R' N6F' R B' R D' SR' F' D' N5L D F SR D' F' N5L' F D2 B D2 B' N5D' B U' D2 B' N5D B U B' D' L' D N5L D' L R D N5L' D' R' N5R' D R2 D' N5R D R N5F' R B R' N5F R B2 N5B' U2 L N5D L' U2 B L B' N5D' B L' N5B SR' B F R' N4F' R F2 B' R' N4F R F L' N4F L' N4F' L R F2 L' N4F L N4F' F2 B U2 L N4D L' D U2 L N4D' L' D' B' U' N4L' U N4D L U' L' N4D' L N4L U L' N4U U N4L' U' L' N4U2 R L U L' R' N4U2 R L N4L U' R' N4U' R D R' N3D' R D' R' N3D N3F D R N3B2 R' U B U' R N3B2 R' N3F U B' U' N3F' D' N3F' L2 B' L' B N3L2 N3B' B' L B L' N3B L' N3L2 R' D' F N3D' F' U2 D F N3D F' U2 R D' L D N3R D' L' D N3R' L' U F' N3U' F U' F' N3U F L NF' L' B L NF L' B' U TL' U' D' L2 D U NL U' D' L2 D NL' U L U' NL SR' D R D' NB NR' D R' D' NR D' NB L' NB' L D R F L' NB L NB2 F' L NB L' B L NB' L' B' D F' D' SF L' F' L2 D' L2 D2 F D2 L D B U B2 U L' F2 L U' B2 U L' F2 L U2


""".strip().split()

# move_map = get_move_map(n)
santa_solution = []
# santa_solution = center_orienting_seq
for move in sse_scramble:
    # print(move, "\t", sse_to_santa[move])
    santa_solution.append(sse_to_santa[move])

santa_solution = ".".join(santa_solution).split(".")
# santa_solution = invert(santa_solution)

if partial_sol:
    santa_solution = partial_sol + santa_solution

print(".".join(santa_solution))

print(f"Validating")
state = initial_state
for move_name in santa_solution:
    state = state[moves[move_name]]

num_difference = evaluate_difference(state, solution_state)
wildcards = puzzle['num_wildcards']

print(f"Num difference: {num_difference}")
print(f"Num wildcards: {wildcards}")


if num_difference <= wildcards:
    print(f"Solution is valid. Diff to WC: {num_difference} <= {wildcards}")
    # Write it to the solution file
    if len(santa_solution) < len(current_solution):
        print(f"New solution is shorter than current solution. Writing to file.")
        print(f"Length of new solution: {len(santa_solution)}")
        print(f"Length of current solution: {len(current_solution)}")
        with open(f"data/solutions/{args.id}.txt", "w") as f:
            f.write(".".join(santa_solution))
    else:
        print(f"New solution is longer than current solution.")
        print(f"Length of new solution: {len(santa_solution)}")
        print(f"Length of current solution: {len(current_solution)}")

        # cube_scramble = " ".join(list(map(lambda x: move_map[x], santa_solution)))

        # print(cube_scramble)
        # print(f"Length: {len(cube_scramble.split())}")
else:
    print(f"Solution is invalid. Diff to WC: {num_difference} > {wildcards}")
    print(f"Expected: {solution_state}")
    print(f"Got: {state}")
    print(f"Writing to partial solution file")

    with open(f"data/sse_partial_sol.txt", "w") as f:
        f.write(".".join(santa_solution))

    cube_scramble = " ".join(list(map(lambda x: move_map[x], santa_solution)))

    print(cube_scramble)
    print(f"Length: {len(cube_scramble.split())}")

