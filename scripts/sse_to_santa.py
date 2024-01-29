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
MR MF MR' MF' MR' N3L' N4U' N5B N6R N7U U2 B2 MR N9B MR' N9B' B2 MR' U' B N9R' B' U MR U' B N9R B' U' F N9D' MF' N9D F' N9D' MF B' N9D N9L' MF' N9L N9D' B N9D N9L' MF N9L R MU' N9R' D N9R' MU N9R D' R' N9R2 B' N9U B N9R' N9D' B' N9U2 B N9D N9R' B' N9U B N9D R' N9D' N9R N9L N9D R N9L' N9D' R2 B N9D N9L N9D' N9L' B' R2 N9R' U' N9L N9U' N9L' U N9L R N9U' N9L' N9U R' N9U N9R2 N8F' N9R' B' N9R N8F N9R' N8U' B N8L' D N9R D' N8L B' N8U B D N9R' D2 U' N9L U D N8L D' N8L U' N9L' U N8L2 N9U' N8L D N8L' D2 N9F D2 N8R2 D2 N9F' D2 N9U R N9U' N8R2 N9U R2 F N7U N7R' F' L' N9U' L F N7R N7U' F' L' N9U L N7L R N9D' N7L' D' N7L N9D N7L' N9L' D N7F' N9D N7F' D' N7F N9D' N7F' N9L D N7F2 N7D2 N9F D N9F' N7D2 N9F D' N9F' L U N7R N9F N9U' N7R' U' N7R U N9U N9F' U' N7R' L' B F N6F' D N9L' D' N6F N6R D N9L D' N6R' F' B' N9R D N9F D' L' N6B2 N6R' L D N9F' D' L' N6R N6B2 N9R' N6F' N9R' N6F N9R L F N9R' N6F' N9R N6F F' N9R' N6U N9B' L N6U' L' N9B L D N9R N6U N9R' D' N9R L' N6U' N9B U' N5B' U N9F U' N5B N9B' N5U N9B U N9B' N5U' N9F' N5R' N9R' B' N9R L' N5F' N9R' N5F N9F' L N5F' L' N9F L B N9R N5F D N9L' D' N9L N5R N5B' N9L' D N9L D' N5B N5L' N9F N5L B' N5L' N9F' N5L B N5D F N9U F' N9U' N5D' N5L2 N9U F N9U' F' N5L2 N4B L F N9D F' L' N4B' N4U L F N9D' F' L' N4U' B' N4L B N9U2 B' N4L' N9U' N4B' N9U B N9U' N4B N9U' F' N9U' F N4R F' N9R F N4R F' N9R' N9U F N4R2 N9F N4D' N9F' U N9F N4D N9F' D N9U N9R2 U' B' N3L' B U N9R2 N9U' U' B' N3L B D2 N3R N9D N3R' D N3R N9D' N3R' N3L U N3L' N9D N3L U' N3L' N9D' U' N9F N3D2 N3B' N9F' U N9F U' N3B N3D' U N9F' U N9F N3D' N9F' N3B' U' N9B U N3B U' N9B' N9L' B ND NL B' N9L B NL' ND' B' N9B' U NR' U' N9B NB N9U' NB' U NB N9U NB' NR U' R' NU R N9F2 R' NU' NB' R N9F' R' NB R N9F' NB' N9R' U2 NF U NL' U N9R' U' NL U N9R2 U2 NF' U2 NB F N8D R F MU' N8F' MU N8F F' R' N8D' F' MF N8U' F N8U MF' N8U' F' N8U L' MR B' N8B' MR' N8B2 MR B MR' N8B' L R2 MF N8D' D MF' N8D2 MF D' MF' N8D' R2 N9R N8B N9R' D2 R' N9R N8B' N9R' N8B R D2 N8B' F N8L' N8F' U N9L U' N8F N9U N8L U N8L' N9U' N9L' N8L U' F' N8R U N9F' U' B N9R B' N8R' B N9R' B' U N9F N8B' N9U' N8B U' N8B' N9U D N9L' N8B N9L N9U F' N9L' N8B' N9L F D' N8B N9U' N8U N8B N8U' B' N8U N8B' N8U' B N8U N8L' N8D L N8D' N8L N8R N8D L' N8D' R U N8U' N8L' N8U N8L U' N8R' R' N8L' F N8U' N8L N8U N8B N8L' F' N8L N8B' N8D N8U' U2 N8B N8L N8B' N8L' U2 N8D' N7F' R N8R N7F' R' N7F N8R' N8U2 N7F' R N7F N7B2 R' N8U2 R N7B2 R' N7F N7B D' N7F2 D N8B2 D' N7F2 N8R D N7B D' N8R' D N7B2 N8B2 N7L2 B N8U B' N7L2 B N8U' N7U' N8F' N7U B' L2 N7U' N8F N7U N8F' L2 N8F N8L' N6U' L' N8L' D N8L N6U N8L' D' N8L' N6U' L N6U N8L' N6F' N8L R' N8F R N6F R' N8F' N6F' N8L' N6F R B' N6L' N8D L B' N8D' N6L N8D N6L' B L' N6L N6F N8D' B N8D N6F' N6D F N8U' F' N6D' N8R N6B N8R' F N8R N6B' N8R' N8U F' N8D' N5F2 N5R D N8L' D' N5R' N5F' N5R' D N8L D' N5R N5F' N8F' N5L' B N5L R' N8B N5L' N8B' R N8B N8U B' N5L B N8U' N8B' B' N8F N8L' B N5L B' N8L B N5L' U' N8U' N5B N8U N5B' U B2 N5B N8U' N5B' N8U B U' N4R N8U N4R' U N4R N8U' N4R' N8F N4R R B N4R' N8F' N4R N8F B' R' N8F' N4R' N4F' D N4F D' N8F N8R' D N4F' D' N4F N8R N8F' N8B N4L N8B' R' N8B N4L' N4B' N8L' N4B R N4B' N8L R' N8B' R N4B F' N8U' N8B L' N3F L N8B' N8U L' N3F' L N3R' N8B N3R F N3R' N8B' N3R B' N3R' B N8D2 B' N3R N3B' D' N3B N8D' N3B' N8D D B N8D' N3B N8D' N8L N3U' N8L' D' N8L N3U N8L' D N3R' R' N8F' N8U2 R F N3U F' R' N8U2 N8F R F N3U' F' N3R D' ND F2 ND' N8L' ND N8L F2 D2 N8L' ND' N8L D' N8D' U NF U' NF' N8D N8L2 NF U NF' U' N8L2 N8B L' ND NF' L N8B' L' U N8B U' NF ND' U N8B' U' L N7F' N7U' R' N7U MR N7U' R N7U F' N7R F MR' F' N7R' N7F' F' L' B' N7L MF' N7L' B N7L' MF L N7L N7F' F' N7U' F' MU F N7U F' N7F' U2 N7F MU' N7F' U' N7F' N7L MU' N7L' N7F U' N7F' N7L MU N7L' N7F MF N7L' B N7L MF' N7L' B' N7L N9L N7B' B N7R N7U' B' D' N9L D B N7U N7R' B' D' N9L' D N7B N9L' N9F N7L' N9F' N7L L U F' N7L' N9F N7L N9F' F U' L' B' N9U N7F2 N9U B N9U' N7F2 N9U N7D B' N9U2 B N7D' N8U' N7F2 R N8U' R' N7F2 N7U R N8U R' N7U' N8U B' N7L B N8L' B' N7L' B N8L N7D' L N7D' N8L' N8U N7L' N8U' L' N8U L N7L N7D L' N8U' N7D' N8L N7D N8L N7D L N7D' N8L' N7D L' N7R N8D' R N8D N7R' N8D' N7B' R' N8D2 R N7B R' N8D' D' N7B' L' N7F N7D' N7F' N7D L N7B D B' N7B2 R' N7U N7R N7U' R N7U N7R' N7B2 N7U' B N7R' N7U' N7R N7B' N7R N7B N7R' D N7R' N7U N7R D' U' N7L' U N7U N7F U' N7L U N7L' N7F' N7U' N7L N7U' L N7U' L' B' N6F2 N6D B L N7U L' B' N6D' N6F2 B N7U N6B' R' N7B R N6B R' N7B' R N6B' N7U' N6B D' N6B' N7U N6B' D N7R' D' N6B2 N6L' D N7R D' N6L N7L D N6L' D' N7L' D N6L U N7B' N6U' N7B N6U U' F' N6U' N7B' N6U N5B F N7R' D N5B D' N7R N7B D N5B' D' N5B' N7U N5U L N7L N7F L' N5U' L N5U N7F' N7L' N5U' N7U' N5L N7U L' N7U' N5L' R' N7F' N5U B N5U' N7F N7D N5U B' N5U' N5R2 B N7D' B' N5R2 R2 B' N7U N7L' B N5U' B' N5U N7L N7U' N5U' B N5U R' N4D N7R' U N7R N4D' N7R' U' N7R R' N7F N4L N7F' R N7F N4L' N7F L' N4B' N4U' L N7F L' N7F' N4U N4B N7F L N7F' U' N4B' N4R' U N7F U' R N7F' R' N4R N4B R N7F R' U N7F R' N4U N7R N4U' R N4U N7R' N4U' D' N3L N7D N3L' D N3L N7D' N3L' N7U2 F' N3R' N7F N3R' F N3R N7F' N3R' N7U2 F' N3R F N7U F' N3R N3D' F N7U F' N3D F B N7U' N3B' N7U B' N7U' N3B N7U' N3F L' N3F L D N7B2 D' L' N3F' N3L L D N7B2 D' N3L' N3F' NL F NL' N7B N7R2 NL F' NL' F N7R2 F' U N7B' NU N7B U' N7B' N7R' NF' N7R' D R N7R NF N7R' NF' R' D' NF N7R2 NU' N7U' NR D L' NR' N7U NR N7U' L D' N7U NR' N7B NF' D' NF D N7B' N7D' D' NF' D NF N7D N6B' U N6B MU' N6B' U' N6B U2 MU2 N6B F MU' N6B' MU F' MU N6B' MU2 N6B U2 D F' N6R' F D' MR D F' N6R F D' MR' B' N6D' F N6D N6R' MF' N6R N6D' F' N6D N6R MF N6R' B D N6F N9U N6F' D' F' N6U' F N6F N9U' N6F' F' N6U F N9F' U N6D N6B' U' N9F U N9F' N6B N6D' N9F U' N6F' N9R N6F R' N6F' N9R' N6F R F N6R F N6R' N9F' N6R N9R F' N6R' F N9R' F2 N9F N9L' N6D' N9L U' N9L' N6D N9L U N8L' N8D' N6B' U' N6B N8D N8F' N6B' U N6B N6R2 U' N8F U N6R2 B' N8L N6B N6D' N8L' B N8L B' N6D N6B' B L N6L D' F' N8F N8R F D N6L' D' F' N8R' N8F' F D L' N7R' F N6L' N6U F' N7R F N6U' N6L F' N7U' N6L U N6L' N7U N6L U' N6L' N7D' N6R U' N7L U N6R2 U' N7L' N6R N7D N6R' U N6R N7F' N7L N6F' R D' N6F N7L' N6F' N7L D R' N7L' N6F N6U' R U2 N6U N7F N6U' N7F' U2 R' N7F N6U N6D' N6F' N6U N6R' N6L N6U' N6F R N6F' N6U N6L' N6R N6U' N6F R' N6R' U' N6R N6D N6R' U N6R U2 N6U N6R D N6R' N6F N6R N6F' D' N6R' U2 N6U D R' N6U' N6L N6U N6L' R D' N6L N6U' N6L' N6U' N5D' N6R N5D L N5D' N6R' N5D N6D' B N6D N5F' N5U N6B' N5U' B' N5U B N6B N6D' B' N5U' N6D N6R' N5F L' N5F' N6R N5F F R' N6U' R N6U N5U N5F' N6U' R' N6U R N5F N5U' N5B R N6F' R' N5B' N5U R N6F R' N5U' F2 N4R' D' N6R' N6F D N4R D' N6F' N6R D F N4L N6B' R' N6B2 N4L' B N4L N6B' N4L' B' N6B' R N6B N6R' N4D L' N4D' N6R N4D L N4D' N4L' U' N4L N4B' U' N4B N6U N4B' U N4B N4L' U N4L N6U' N3R N6F N3R' F' N6F' L' N6F N3R N6F' L N3R' F L D' N3B N6B N6U' N3B' D N3B D' N3B N6U N6B' N3B' D N3B' L' D' N6F N3U' N6F' D N6F N3U U N6R' U' F' N3F N3L' F U N6R U' F' N3L N3F' F N6R F NR F' N6R' N6U F NR' F' N6U' N6F' D' N6B NU NF2 N6B' D N6B D' NF2 NU' D ND' N6B' U N6B ND N6B' NR U' N6R' U NR' U' N6R N6U NB' NL U NB L N6F2 L' NB' U' NL' NB U L N6F2 L' U' N6U' L' B MU N5B U' N5B MU' N5B' U B' N5B' MF' N5R F' N5R MF N5R' F L N5R' B' F' N5D L N5D MR N5D' L' N5D' MR' SF MR N5U MR' N5U' B2 R N5D R' N9U R N5D' N9U' N5L N9U R' N5L' D' N5L N9U' N5L' D N5R R N9F N5U N9F' N9L N5U' N9L' R' U' N9L N5U N9L' N9F N5U' N9F' U N5R' N9R R N9D' R' B' N5B' N5D B R N9D R' B' N5D' N5B B N9R' N5R' U' N8L2 U B N5R' B' U' N8L2 U N8U2 B N5R B' N8U' N5R U N5R' N8U' N5R U' N5U' N8L' D N5B' D' R' N8F2 R D N5B N5D' D' R' N8F2 R N5D N8L N5U N5D N8R N5D' L' N5D N8R' N5D' L U' N5R N7B N7L N5R' U N5R U' N7L' N7B' U N5D' B N5D N7F N5D' B' N7F' D N7F N5D N7F' D' N5R2 D N5R N7U N5R' D' N5R N7U' D N5B' N7D' N5B B L' N5B' N7D N5B N7D' L B' N7D N5U B N7L B' N5U' N5L' B N7L' B' N5L D' N5B2 N5D' R' N6F' R N5D N5B R' N6F R N5B N6L' F N6L N5F N5U' N6L' F' N6L N6D' F N6D N5U N5F' N6D' F' N6D B N6U' N5F N6U B' N6U' N5F' N6U L N5D L' N6U' N5L N6U' L N6U N5L' N6U' N5D' L' N6U2 N5D N5R N5D' L N5D N5R' N5D' L' U' R N5U N5R N5U' R' N5U' N5R' N5U N5R N5F N5R' N5F' U F N5U' N5F N5B' U N5F' N5U N5F U' N5U N5B N5U' N5F' F' N5U' N5L U' N5L' N5U N5D N5L U N5L' N5D' N4F' L N5D D R N4F' R' D' N5D' N5F' D R N4F R' D' N5F L' N4F N5L2 U N4L' U' N5L N4U' N5L U N5L' N4U N5L N4L U' N5D L F' N5L N4B' N5L' F L' N5D' N5L N4B N5L2 N5B' N4B' L N4B L' N5B N5L N3U' U' N5U' N5R' U F N3R' F' U' N5R N5U U F N3R F' N3U D' L N5U N5L' L' N3D' L N3D N5L N5U' N3D' L' D N5B N3L N5L N3B N5L' F N5L N3B' N5L' N3L' N5B' N3L F' N3L' N3D N3B2 D N5R' D' N3B' N3L' D N3L N5R N5D N3L' D' N3L N5D' N3B' F' U' NB R' N5B' R NB' ND' R' N5B R ND U F N5R' B N5R B' ND NL' B N5R' B' N5R NL N5F ND' F' ND N5F' ND' F N5L2 F' NU F N5L2 NF N5D F' N5D' NF' NU' N5D F N5D' N4B N4L L D MR N4U' MR' D' MF MR N4U MR' MF' L' N4L' N4B' L N4B' D MR' N4U' L' N4U' MR N4U L D' N4U N4B N4D2 B MR B' MR N4D MR' N4D B MR' B' R MF' N4D F' N4D MF N4D' F N4D' L' R' D2 N9R2 D F N4L F' D' N9R2 D F N4L' F' D N4R' N9U L' N9U' N4R N9U L F N4D' F' N9U' F N4D F' N4B' N9D' B N9D N4B N9D' B' N9D R N9U R' N4U2 R N9U' R' N9R' N4U' N9R B L' N9R' N4U N9R N4U' L B' N4U' N8L2 F N8D N4F2 N8D' F' N8L' N4B' N8D F N8L N4F2 N8L' F' N8D' N4B N8L' N8D' N4F' N8D B' N8D' N4F N8D B N8U L' N8U' N4R' N8U L N8U' N4R R' N4D2 R N8B R' N8B' N4D2 N4L' N8B R N4L B' N4L' N8B' N4L B L2 N7R R N7D' N7F R' N4D R N7F' N7D R' N4D' L2 N4U R' N4U' N7R' N4U R N7U N4L' N7U' L N4L U N4L' N7U N4L U' L' N4U' L N7U' L2 N7D' N4B F' N4D F N7D F' N4D' N7D' N4B' N7D F L2 N4U N6R' N4U' L' N4U U' N6R N6B D' N6B' N4U' N6B D N6B' N6R' U N6R N4R F N4R' N6F' N4R F' N4R' N6F N4D N6L' N4D' R N4D N6L N4D' N4U' N6R' N4U N4F' R' N6U R N4F R' N6U' N4U' N6R2 N4U R' N4U' N6R' N4U N6B' N6D L' N4D' L N6D' N6B N6D' L' N4D L N6D R N4B2 N4R' N4U' F N5R' F' N4U N4R F N5R F2 N4B2 N4L F N5L' F' N4L' F N5L N4U' F N5U F' N4U F N5U' N4D N5B N4D' F' N4L' N5B' D N5B D' N4L N4D D N5B' D' N4L' L' N5F' L N5F N4L N4D' N5F' L' N5F L R2 N4F' L' N4U' N4L' N4U2 N4L N4U' L N4F R2 B' N4B' N4L N4B B N4U' B' L2 N4B' N4L' N4B L2 B N4U N4L N4U2 N4L N4U N4B N4U B' N4U' N4B' N4U B N4L' N4U' N4L' D' N4L N4U N4L' D N4B' N3R' L2 N4L2 D' N3R N4B N3R' N4B' D N4L2 L2 N4B N3R N4R' N3D' N4R D N4D B N4R' N3D N4R N3D' B' N4D' D' B' N3F2 L N4U' L' N3F2 N3D' L N4U L' N3D B N3D N3R N4B2 D N3L N3B D' N4B2 D N3B' N3L' D' N3R' F NU' NL' U B N4R' B' U' NL NU' U B N4R B' U' NU2 F' N4R2 L' N4F L NF L' N4F' NF' N4R NF L N4R F' N4R' NF' N4R F N4B' NR N4B NR' B D' NR N4B' NR' N4B D B' NB F' N4D F U N4F ND' N4F' U' N4F F' ND N4F' ND' N4D' F ND NB' N3R' F' D' MR' N3D R N3D MR N3D' R' D N3D' F N3R D' B N3D' L' N3F L N3D MR N3D' L' N3F' L N3D MR' B' D U R2 N3D MR N3D' R N3D' MR' N3D R N3F' MR' N3F MR N3F U' MF U N3F' U' MF' B2 N3B' D' N9R U2 N3B' U2 N9R' U2 N3B U2 D N3B N9F N9D F N3R' F' N9D' N9L2 F N3R F' N9L2 B2 N9F' L N9D' N3R' N9D L' N9D' N3R N9D R' F N3U' F' R' N9D2 R F N3U F' R' N9D2 R2 L N8U' N8R' B' N8R N3R N3B' N8R' B N8R N3B N8U B' N3R' B L' U' N8B' N3D' N8B U N8B' N3D N8B B' N8R' N3F' N8R B N8R' N3F N8R N3B' R N3B N8L N3F N8L N3F' N3B' R' N3B N3F N8L' L' N3F' N8L' N3F L N3F' B' N3R B N7U2 B' N3R' N7U N3B N7U' B N7U N3B' N7U N3D L N7D' L' N3F N3D' L N7D2 L' N3D2 L N7D' L' N3D' N3F' N3D' D' N3B N7R N7F N3B' D N3B D' N7F' N7R' D N3B' F' N7U' N3B U' N3B' N7U N7L2 N3B U N3B' N3F' U' N7L2 U N3F SF N3U N6F' N3U' B N3U N6F N3U' N6R' B' N3R B N3R' N6R N6B N3R B' N3R' B N6B' U' N6B N3U N6B' U N6B N3U' N6B' N3L N6D L N3F2 L2 N3D' L N6D' L' N3D L2 N3F' L' N3F' N6U N6B' N3F L N3F' L' N6B N6U' N3L' N5L2 D N5L' N3U N5L D' N5L' N3U' N5L' U N3D N5B' N3D' N5B U' R' N5B' N3D N5B N3D' R N3D L' N3F' N3D' L N5D L' N3D N3F L N3D' N5D' N3R' D' N5L' D N3R D' N5L N3R N5U' N3R' D N3R N5U N3R' N3L N3U2 N3L F' N3L' N5R N5F N3L F N3L' N5F' N3U2 F' N5R' F N3L' N3D' R N3F' L' N4D L N3F N3U' L' N4D' L N3U R' N3D N3B F' N4U N3F' N4U' N3U' F N4U F' N3U F N3F N4U' N3B2 N4B' L' N3U' N4L N3U L D' N3U' N4L' N3U N4L D N4L' N3B N4B N4F R' N4U N3R' N3U' N4U' R N4F' N4U R' N4F N3U N3R N4F' R N4U' N3F' N3R' F N3U B' N3L' N3U N3L N3U' N3F B N3U' N3F' F' N3R2 L' N3F L N3U L' N3F' N3U N3R' N3U' L N3U' N3F N3B N3L B D' N3D N3B' N3D' N3B D B' N3L N3B' R N3B' N3L2 N3B R' B' U' N3U' N3L U B NL' B' U' N3L' N3U U B NL ND N3B U' NR U N3B' N3L U' NR' U N3L' ND' R NB N3R NB' R' NB N3R' NB' N3D' NR' N3D F' N3D' F NR ND F' N3D F ND' TF' NR' D MF' D' NR D NR MF NR' D' TF NL MU' NL' U NL MU NL' NU MF R MU' R' NU2 R MU R' NU U' NF' U MF' U' F' R MF2 ND' MF' ND MF' R' TF NR N9B' U2 N9R2 U NB' U' N9R2 U2 N9B U' NB N9L U NR' U' N9L' NF N9D NF' U' N9F' U' NF U N9F U' N9D' NF' U' ND' N9B ND B' ND' N9B' ND B N9B' D' N9B TD NL' D' N9B' D N9B NL ND' F2 N8U' NB N8U F2 N8U' NB' F N8U NF' N8U' F' N8U NF TD' NB' D N8B' D' NB TD B' ND' N8B ND NL N8F' NL' B NL N8F N8U L' NF TL N8F' L' N8F NL' N8F' NF' L N8F N8U' NL' R' NR NB L' U' N7F U L NB' NR2 L' U' N7F' U L R U' N7L' U NR U' N7L U N7L' N7U' D' NF' D NF N7U N7L NF' D' NF D N7U' L' NF N7R' NF' L F' NF N7R NF' N7R' F N7R N7U NL' N7D' NL D' NL' N7D NL D N6F' N6D L' NF L N6D' N6F L' NF' L B' N6L ND2 NB' N6L' B' N6L TB ND2 B' N6L' N6R' B' NR' B NF N6R B' N6R' NF' N6R NR N6L' B' NL B N6L B' NL' N6D' N6B' R' N6D D2 NR' D2 N6D' R N6B N6D D2 NR D2 N5L NL2 D N5F D' NL2 NF2 D N5F' D' NF2 B NU2 N5B NR B' NR' N5B' N5L' NR B NR' NU2 B' N5U' N5B NU' F L' NU N5B' NU' N5B L F' N5B' NB R NB' R' N5U N5R' R NB R' NB' N5R NU N4F' D' NF D NF' N4F N4L' NF D' NF' D N4L NB' U R' NB N4L' NB' N4L R U' N4L' NB N4L R' NB R N4B' NB' N4R NB R' NB' N4R' R N4B' R NF R' N4B2 R NF' R' ND' L ND N4R ND' L' ND N4R' NB' D' N3F' R NB' R' N3F N3D' R NB R' N3D D NB U B' NU N3F NU' B U' NF' TU N3F' TU' NL N3F R N3F' NL' VF N3L NF' R' NF N3L' N3F' NU N3B NU' L F' NU N3B' NU' N3B F L' N3B' U NB2 ND NB U' NB' ND' NB NR' U NB U NU2 TR' NU NR NU' R TU2 NR ND U NF' ND' F' ND' NF ND F U' ND' NR' B ND NB ND' B' ND NB' ND NR ND' L U' L' MU' L MU U F2 MU' L' MU L F2 L' B' D' MF' D B R2 D' R2 MF R2 D R2 L U MF U' L F' U L' MF' L U' F L2 N9R' R' F N9D' F' R D' F R' N9D R F' D N9R D' L' N9B' L B L' N9B L B' L' N9D' B L' N9F L SF L' N9F' L F' N9D L D F R U N9B' SU B2 SU' N9B U N9F' D' B2 D N9F U2 R' N9B F' N9F' L2 N9B' N9F' R N9F N9B L2 N9B' N9F' R' N9F2 U R' U' N8L U R U' N8L' D N8F D' L' D B' L N8F' L' B D' L N8R R D' B' N8D B D U2 B' N8D' B U2 N8R' R F' N8R F R2 D' L F' D N8R' D' F L' D N8B U' F U N8B' U' F' U L' R' N8F' R F' D2 R' N8F R N8F' D2 F N8F L SU2 L D2 SR N7D2 SR' D2 SR N7D2 R' SU2 N7U' F' U N7U B N7U' B' U' F U' B N7U U N7B' U' B' U N7B U N7B U' B2 U N7B' U' B' U R' N7B' R N7F U' B' U N7F' R' N7B R U' N7U R L N7F2 L' R' N7R' F' R F N7R F' L N7F2 L' R' F N7U' F N6L2 F' L2 F N6L2 F' L U N6R' U' L U N6R U' N6B2 R' F R N6B2 R' U' N6L N6F' U F' U' F N6F N6L' F' U R F L' F' N6R' F L R F' N6R F R' F' L2 B' L' N6D L B D2 L' U B' N6D' B U' L D2 L2 U' R F' N5R' F R' L2 F' N5R F L2 U R D' N5F' D N5R B' D' R B D N5R' D' B' R' N5F D B SR' D L' N5U L D' L' N5U' B N5B N5F N5R2 D' N5R2 B2 N5R2 D N5R2 N5F' N5B' D' B2 D B' R' F' N5L F R' F' N5L' F R2 N4R' U' L' U N4R U' L U N4R N4D U' L' D' N4D' R' N4D R D L U R' N4D' R N4R' N4B' U' B' N4U2 F' N4U B N4U' F N4U U B' N4U B N4B D N4F' D' L' D B' L N4F L' B D' L F L' B' L N3F R B N3L' B' R' L' B L N3L N3F' R' N3R' N3F' R F' R' F N3F N3R F' R N3D U R F' R2 N3D' R2 F R' U' R' F' N3D F R N3D' R' D R D' N3R N3B D R' D' R N3B' N3R' F U' F NU F' U F NU' F2 D F2 NL2 F' R2 F NL' F' SR F NL' F' L R F' D' B D R' B' NR B NR' R D' NR B' NR' R F' R' NB R F R' NB' F' D' U' F R' NU' R F' U D F R' NU R L2 U' B2 U L' F2 L U' B2 U L' F2 L'


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

