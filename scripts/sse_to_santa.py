#!/usr/bin/env python3

import argparse
import pandas as pd
from util import *
import re

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("--partial_sol", type=str, default=None)

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
    print(f"Applying partial solution: {partial_sol}")
else:
    partial_sol = None

# if n % 2 == 0:
#     center_orienting_seq = []
# else:
#     initial_state, center_orienting_seq = orient_centers(initial_state, moves, n, solution_state)
#     print("Center orienting seq")
#     print(center_orienting_seq)
#     scramble += center_orienting_seq

move_map = get_inverse_move_map(n, False)
# print(move_map)
cube_scramble = " ".join(scramble)
cube_scramble = list(map(lambda x: move_map[x], cube_scramble.split()))
cube_scramble = " ".join(cube_scramble)
print("\nCube scramble")
# print(cube_scramble)
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
MF' N3L' N4U' N5B N6R N7U D MR U' N9U2 MR N9B' MR' N9B U MR' N9U2 N9L D' MR2 D N9L' D' MR2 N9L' D2 MF D2 N9L' D2 N9L MF' N9L' D2 N9L2 MU F L' N9L2 MU N9L MU' L2 MU N9L MU' L' F' MU' N9L B2 U2 L N9U N9B' N9U' N9B N9L' L' N9B N9L N9B' U2 B2 N9L' D' R' N9D' N9B N9D N9L2 N9B' R N9B N9L2 B N9D' N9B' N9D B' D U N9B' N9D2 N9B U' N9B' N9D2 N9B' N9L' N9B2 B' N9R D' N9R N9U' N9R' D N9R' N9U B N9L N9D B2 N8R B2 N9D2 B2 N8R' N8L N9B N8L' B2 N8L N9B' N8L' N9D D' U' N9L U D N8L D' N8L U' N9L' U N8L2 N9U' N8L D N8L' N9U F N8R2 N9U' L' N9U N8R2 N9U' N8D L N9U N9F2 L' N8D' L N9F2 F' N9D N7L2 B2 N9U' B2 N7L2 B2 N7R N9U R2 N9U' N7R' N9U R2 B2 N9D' N7D B' N9L' B N7D2 N7L' B' N9L B N7L N7D N7L N9D' N7L' D' N7L N9D N7L' D B2 N7R N9B' N7R' B2 N7R L2 N9B N7R' N9B' L2 N7D' N9B D N9B' N7D N9B D' F' B' N6D N9B N6D' B N6D D2 N6L' D2 N9B' D2 N6L D2 N6D' F N6F' N6D SU2 N9R U2 N6F U2 N9R' U2 N6F' D2 N6D' N6F N6U' N9F' D' L N9F L' N6U N6B L N9F' L' D N9F D' N6B' D N6L2 N6D' N9D N9R' N6D L N6D' N9R N6B2 L' N9D' L N6B2 N6D L' N6L2 N5B' L' N9B' N5B' N9R2 N5B L N5B' N9R2 L' N9B L B' D' U2 N5R' U2 N9B' U2 N5R U2 N9B2 D N5B2 D' N9B' D B N9U N9R' L2 N5F L2 N9R F N9U N9R' L2 N5F' L2 N9R N9U' F' N9U' L' B' N5L' N9F' N5L B N5L' L2 N5D L2 N9F L2 N5D' L' N5L N9F N4D' N9F' U N9F N4D N9F' SU N9L' N4D' N4R' N9L D N9L' N9F2 D' N4R D N9F2 N4D N9L U2 N9B R N9B' N4R N9B R' N9B' N4R' N9L B' N9B' D' N9B N4U N9B' D N9B F2 N9L F2 N4U' F2 N9L' F2 B N9L' L' N3D N9L' U N9L N3D' N9L' U' N9L N3B' L' N9B N3B N9L N3B' L N3B N9L' L' N9B' N3L' N9B' N3D' N9R N3D L2 N3D' N9R' N3D N3U2 L2 N9B L2 N3U2 N3L N9L R' N9D' R N9D N3D N3B' N9D' R' N9D R N3B N3D' N9L' N3U N9R2 N3U' F' N9U N3U N9F N3U' F N3U N9F' F' N9U' F N9R2 N3U' F' N9D F ND2 F' N9D' ND' N9B' ND F ND' N9B ND' NB' N9L NB R2 NB' N9L' NB SR2 N9B' L2 NU B NU L2 N9B L2 NU' B' TU' N9R NB2 ND' N9R' U N9R U' ND NB2 U N9R' D2 F N9L' B' NR' ND' B N9L B' ND NR SF' D2 MF' R' MF N8R2 MF' R N8U MR N8U' MR' MF N8R2 N8D2 D2 F N8D MR N8D' MR' F' D2 MR N8D MR' N8D L' N8U' F U N8L' MU2 N8L MU2 U' F' N8U L N8D' B' N8L MF' B MU B' N8L' B MU' MF N8D N9F B N9D2 B' N8L' N9F N8D' B N8D N9F' N9D2 N8D' B' N8D N8L N9F' N8L2 N8B N9D N9L' N8B' U' N8L' N8B U N8L N9L N9D' N8L' U' N8B' N8L U N8L2 SF SR N8D' N8F' L N9D' L' N8F N8D L N9D R' SF' U' N8F U N9D2 N9R2 U' N8F' U N8F N9R2 N9D2 N8F' N8D U2 N8B N8L N8B' N8L' U2 N8D' F N8D2 F N8R F' N8D2 F N8R' N8L2 N8U2 N8L' N8U' R' N8U N8L' N8U' R N8U' F2 N8B N8U' F' U2 N8R' N8U2 N8R U2 N8U' N8F N8U' F N8U N8F' N8B' U' N7R2 U N8B2 U' N7R2 U N8B2 B N7U' N8F' N7U B' L2 N7U' N8F N7U N8F' L2 N8F L N8F U2 N7L U2 N8F' R' N8F' U2 N7L' U2 N8F SR N8D' F N8D N7B N8D' F' N8D N7B' N8D' B' N8D N7F N8D' N7U' B N8D2 B' N7U B N8D' N7F' B N6R' N8D2 N8B N6R F' N6R' N8B' N6L2 F N8D2 F' N6L2 N6R SF N6R2 B' F2 N8F2 N6L N8F2 F2 N8D' F2 N8F2 N6L' N8F2 F2 N8D B N6R2 N6F' L N6F N8L2 N6F' L' N6F N8L R N6U N8R N8D' N6U' R' N6U R N8D N8R' R' N6U' N8L SF N5D2 B N8L2 N5D' N8F N5D B' N5D' N8F' N5D' B N8L N5F N8L F' N8L' N5F' N8L N8R2 D N5F' N8U' N8R2 N5F D' N5F' N5B D N8R2 D' N5B' N8U N5F N8R2 N8U' L' N8B' N8U2 L B N5U B' L' N8U2 N8B L B N5U' B' N8U N4R' B' N8R B N4R B' N8R' B' N8R N8F' F2 L N4D L' F2 N8F N8R' F2 L N4D' L' SF2 N4R' B' N4U' N4B' B L N8D' L' B' N4B N4U B L N8D N4R N4F' L N8R N8D' L' N4F L N4F' N8D N8R' N4F L2 U' SF N3R' F2 N8U F2 N3R F2 N8U' F B N3B' N8D' N3B U N3B' N8D N3B N3U2 L' N8U L N3U2 L' N8U' L N8F N3R N8F' L' N8F N3R' L' N3B' L N8F' L' N3B N8R' B' N8R N3D' N8R' N3D B L2 N3D' N8R N3D' N8R' U N8R N3D2 N8R' U' N8R NL' N8L' F' N8L NL ND N8L' N8F ND' NL' F NL ND N8F' U' N8L ND' N8L' U N8L L2 D' N8B' NF ND N8B D N8B' ND' N8L2 D' NF' D N8L2 N8B L2 NR' D' NR N8D NR' D NR N8D' N7F2 L' MU N7L2 MF N7L' MF' N7L' MU' L N7F2 N7L' D' MF' D N7L D' N7L MF N7L' D SF' MU B' N7R' B MU' B' N7R U' N7R' MU2 N7R U N7R MU2 N7R' F MU' N7B' MU' B' MU N7B2 MU2 N7B' MU B MU2 N9L N7B' B N7R N7U' B' D' N9L D B N7U N7R' B' D' N9L' D N7B N9L' N9F N7L' N9F' N7L L U F' N7L' N9F N7L N9F' F U' L' B' N9U N7F2 N9U B N9U' N7F2 N9U N7D B' N9U2 B N7D' N8B N7R N8D' N7R' N8B2 U2 N7L U2 N8B2 U2 N7L' N7R N8D N7R' U2 N8B' D' N8B' D N7F' D' N8B D N7F N8F N7R D N8B2 N8R D' N7R' D N8R' N8B2 D' N8F' F2 N8R' N7B2 N8R F2 N8R' N7B2 N8R N8L2 F' N7L' F N7L N8L2 N8D2 N7L' F' N7L F N8D2 B R N7U' N7R N7U N7R' R' B2 N7B' U' N7B' N7U2 N7B U N7B' N7U' N7B2 N7U' B N7B' N7D' F' N7U' F N7R2 N7U N7R2 N7U' N7D F' N7U F N7B U' N7L' U N7U N7F U' N7L U N7L' N7F' N7U' N7L N7D2 N6B2 R N7U N7D' R' N7D N6B2 N6L2 N7D' R N7D N6L2 N7U' R' N7D N6F N7D F N7D' N6F' N7D F' N7U2 N7L' U' N7L N7F2 U N6L U' N7F2 N6D2 N7L' U N7L N6D2 N6L' N7U2 N6B2 D N7R' N6B' N7U2 N6B D' N6B' N7U2 N6B' D N7R N6B' D' N7B D N6B D' N7B' F' N5D N7F' F' N7D' F N5D' F' N7D N5D N7F N5D' F2 N7L' N5D2 B' N7D B N5D2 N5R2 B' N7D' B N5R2 N7L N5F2 L' N7F' L N5F2 L' N5L2 N7F L N7F' N5L2 N7F R2 B' N7U N7L' B N5U' B' N5U N7L N7U' N5U' B N5U R' N5B' R' N7B R N5B R' N7B' L2 N4B N7L' N4B' L2 D2 N4B N7L N4B' N7L' D2 N7L N4R' B N4R' D2 N7F' D2 N4R B' N4R D2 N7F D' N7F' D' N4B' D N7F D' R' N4F R' N7U N7F' R N4F' R' N4F N7F N7U' N4F' R2 N4B N7B' R N7B N4L' N7B' R' N7B N4L B2 N7L D F N3L N3U2 F' D' N7L' D F N3U2 N3L' F' D' SF2 N3U' N7B N7U' N3U F N3U' N3R2 F' N7U F N3R2 N7B' N3U F N3D2 R N7B2 R' N3D2 R N7B' N3L N7B' R' N7B N3L' N7B' D' N3L N7D N3L' D N3L N7D' N3L' N7R2 F NF2 NU2 F' N7R F N7R' NU2 NF2 N7R F' N7R N7F NR' N7F' L' N7F NR N7F' NF' L N7U L' NF L NR N7U' L' N7U NR' N7U' L F ND' N7B ND F' ND' N7B' N7L' ND R2 F2 ND' N7L ND N7L' F2 R2 N7L N7B' ND2 N7B D2 N7B' ND2 N7B D2 MR2 F N6L2 F' MR2 F N6L2 MR' U N6U2 MR N6U2 MR' N6R U' MR U N6R' U' F' U2 N6B MR2 N6B' MR2 MU2 U2 N6B MU2 N6B' R' MU R' MF N6L' MF' R MU' L MU N6L MU' SR D N6F N9U N6F' D' F' N6U' F N6F N9U' N6F' F' N6U F2 N9D F' N6D2 F N9D' F' N9R' N6D N9D R' N6D' R N6R N9D' R' N9D N6R' N9D' N9R N6D R N6D N9F N9R' N6U' N9L N9F2 L2 N6U L2 N9F2 N9R2 L2 N6U' L2 N9L' N6U N9R' N9F' SU N8F' D N8F N6R2 N6B' N8F' D' N8F D N6B N6R2 U' N6R' N8B L' N8B' N6R N8B L N8B' L N6U' L' N8D SR' N8D R N6U R' N8D' SR N8D' N6F' N8D' F' N8D N6L N6F2 N8D' F N8D N6F2 N8U F' N6L' F N8U' N6F N7U' L' N7B2 N6R' N7F L N7F' N6R N6D2 N7F L' N7F' N7B2 L N6D2 N7U N6L2 N7U N7F' R N6D' R' N7F N7U' N7F2 R N6D R' N7F2 N6L2 N7U N6R' N7U' D R2 N7U N6R N7U' N6R' R2 D' N6R2 N7F' F2 N7R' F2 N6U F2 N7R F2 N7F D N7F' N6U' N7F D' N6R' D N6R2 N6B N6L N6D L N6D' L' N6L' N6B' N6R2 L N6D L' D' R N6F R' N6B2 R N6F' R2 N6B N6L N6B' R N6L' N6B2 N6D' L' D' N6L' N6U2 N6L D N6L' L' N6U2 N6F' N6U2 N6F L N6F' N6U2 N6F N6L L2 N5U N6U N6L' N5U' L N5U L' N6L N6U' L N5U' N5L L' N5B L' N6D' N6B L N5B' L' N6B' N6D N5L' N6B L N6B N5R N5D' N6B' L' N6B L N5D N5R' L' N6B2 SU N6L D N5F2 D' R' N6L' N6F' R D N5F2 D' R' N6F R SU' N4B2 N4L2 B' N6U B N4L2 N6B' N4U' B' N4U N6B N6U' N4U' B N4U N4B2 N6F2 N6R' N4L' F' N4L N4D F N6R F' N4D' N6F N4L' F N4L N6F SU' F' N4R' F U N6U N6L' U' F' N4R F U N6L N6U' D' N3D N6L' N3D' F L N3D N6L N3D' N6L' L' F' N6L N6R2 R D N3R D2 N6F D2 N3R' D2 N6F' D R' N6R2 D' N6F N3U' N6F' D N6F N3U U N6R' U' F' N3F N3L' F U N6R U' F' N3L N3F' F N6F' NL2 B N6U B' NL2 B N6U' B' N6D' L' N6F L' D' NB' D L N6F' N6R' L' D' NB D N6R L2 N6D N6F N6R F NR F' N6R' N6U F NR' F' N6U' N6F' D' N6B NU NF2 N6B' D N6B D' NF2 NU' D ND' N6B' U N6B ND N6B' U' MF' N5U B N5U' MF N5U B' N5U' D' MR F D2 MR' N5F' MR N5F D2 N5F' F' MR' N5F D MR' MF' U' MR N5D' MR' U MF N5D MR L N5U2 L N5U MR' N5U' L' N5U' MR N5U' L' N9U N5L' D' N5L N9U2 N9R' N5L' D N5L D' N9R D N9U N9R R2 B N5L N5B' B' R2 N9D R2 B N5B N5L' B' R2 N9D' N9R' R' N9U' N5F2 N9U' N5B F2 N9U F2 N5F2 N5L2 F2 N9U' F2 N5L2 N5B' N9U2 R N8F N8L' F' N8U N5L' N8U' F N8L F' L N8U N5L N8U' L' F N8F' N5R' B' N5R B N8F2 N8R2 B' N5R' B N5R N8R2 N8F2 N8B' R N8B N5R' N8B' R' N8B N5R' N8D' N5R D' N5R' N8D N5R N8L' D N5R N5B2 D' N8L D N5B2 N5D2 N7L U' N5D' R N5D N7L' N5D' R' N5D' N7L U N7L2 N5B' R N5B N7L N5B' R' N5B R L2 D' N5F D L2 N7D' L2 N7F' D' N5F' D N7F L2 N7D R' B N5L N7F N7U2 N5L' B' N5L B N7U2 N7F' B' N5L U' N7B2 U N5L2 U' N7B2 U N6R N5F2 N6U F' N5L2 F N6U' N6L' F' N6L N5L2 N5F2 N6L' F N6L L N6R' N5D L' N6U' N5L N6U' L N6U N5L' N6U' N5D' L' N6U2 N5U D2 N6F D2 N5L D2 N6F' D2 B' N6R B N5L' B' N6R' B N5U' L' N6F L N5B L' N6F' L N5B' N5D' F' N5D N5B' N5D' F N5D N5B N5F N5L N5F2 N5U N5R' U' N5F2 N5L' N5F2 N5L U N5R N5U' N5L' N5R' N5D2 D2 N5R D2 N5R' N5D2 N5R N5F D2 N5R' D2 N5R N5B N5R U' N5R2 N5B N5R2 N5B' N5R' N5D' N5R U N5R' N5D D N5B' N5U' N5B D' N5B' N5U N4F N5U2 N5F' U' R' N4B' R U N5F N5U' U' R' N4B R N5U' U N4F' N4L' N5B2 N5D D L N4B' L' D' N5D' N5B2 D L N4B L' D' N4L N4U2 R' N5U2 R N4U' R' N5U2 N4U' N5L2 N4U R N4U' B' N5L N4F' N5L' B N5L N4F N5L F' N3R' R2 N5B' R2 N3R F N3D' N3R' R2 N5B R2 N3R N3D N5L U' N5L' N3U' N5L U N5L' F' N5U F N3U2 F' N5U2 F2 N3R' F2 N5U F2 N3R F' N3U' N5D' D' N3B N3L2 F N5D F' N3L2 F N3B' N5D' F' D N5D N3B N5D' N3B' N5D N5L B N5D' L2 NB' L2 N5D L2 NB L2 B' N5L' NL' U2 N5F' NR' N5F NR2 U2 N5B U2 NR' U2 N5B' NR' N5F' NR N5F NL ND NL' B ND' N5F' N5R' ND B' NL ND' B NL' N5R N5F NL B' N4D' MR' F D' MF' N4R' MF N4R D F' MR N4D2 R N4D' MR' N4D R' N4D' MR SU N4L' MF' N4L MF D N4L MF' N4L' MF2 N4U' MF' U' MF N4U MF' MU' F N4B N4D F' MU F MU' N4D' N4B' MU F' D2 N9R2 D F N4L F' D' N9R2 D F N4L' F' D N9L2 N4U L N4U' N9L2 N4U L' N9R' N4U' L' N4U N9R N4U L N9U' L' N4U2 L N9U N9B N4R N9B' R N9B N4R' N9B' SR' N4R N4D2 L N9D L' N9D' N4D2 N4R' N9D L N9D' L2 N4U' R' L' N4F2 L N8D N8U L' N8U' N4F2 N4R' N8U L N8U' N4R N8D' R N4U R' N4D2 R N8B R' N8B' N4D2 N4L' N8B R N4L B' N4L' N8B' N4L B' N4B N8U F' U2 N8L' D2 N4B' D2 N8L D2 N4B SU2 F N8U' N4B' B2 N7L B' N7U N7D' F N4R2 F' N7D N4F2 N7U' F N7U N4F2 N4R2 N7U' SF' N7L' N4R2 D L2 N4D' L2 N7B L2 N4D L2 D' N4R D N7B' D' N4R N4U N7U N4L' N7U' L N4L U N4L' N7U N4L U' L' N4U' L N7U' L' N6F N4D F N4D' N6F' N4D F' N4D' N6L' L' N6F' L2 N4U' L2 N6F L2 N6L D2 N6L' N4U N6L D2 SR N4D' N6R' N4D R' N4D' N6R N4D2 L2 N4D' N6L' N4D L2 N6L U' N6L' N4D' N6L U F L' N6D' N4R2 N6D L' N6D' N4R2 R2 N4B2 R2 N6D R2 N4B2 SR2 F' N5R' U' N4B2 U N5R' U' N4B2 N4R2 U N5R2 U' N4R2 U2 N5U2 F' N4L F U2 N5U2 N5B U2 F' N4L' F U2 N5B' U N4D N4L' L' N5F' L N5F N4L N4D' N5F' L' N5F L SU' N5F L N4B2 N4U L' N5F' L N4U' N4B2 L' SU N4U R N4U R N4F N4U' N4F' N4U R' N4U' R' N4U' N4L N4U2 N4L N4U N4B N4U B' N4U' N4B' N4U B N4L2 N4F' N4U2 L' N4F' L U' N4U N4F' N4U N4F U L' N4F L N4F N4U B2 N4U' N4F' N4U B2 N4U' N4F N4B' N3R' L2 N4L2 D' N3R N4B N3R' N4B' D N4L2 L2 N4B N3R N4R' N3D' N4R D N4D B N4R' N3D N4R N3D' B' N4D' D' B' N3F2 L N4U' L' N3F2 N3D' L N4U L' N3D B N3D N3R N4B2 D N3L N3B D' N4B2 D N3B' N3L' D' N3R' N4R2 NU' N4R' D' N4R NU N4R' D N4R' NB F' L N4D' L' F' NL NU F L N4D L' F' NU' NL' F2 B' N4R' B U2 NR2 NF U2 B' N4R B U2 NF' NR2 U2 L NB' N4R2 NB L' NB' N4R2 R2 NL' N4B B U2 N4B' NL N4B NL' U2 B' NL N4B' R2 N3L' MU' R' N3U R MU R' U' MR N3U' MR' U R N3L B D' N3L2 D MF' D' N3L2 D N3D MR' N3D' MR MF B' N3B' L MR2 MU2 L' N3B L N3B' MR2 MU2 N3B MF L' N3F N3D' L MF' L' N3D N3F' B2 N3B' D' N9R U2 N3B' U2 N9R' U2 N3B U2 D N3B N9F N9D F N3R' F' N9D' N9L2 F N3R F' N9L2 B2 N9F' N9D2 B N3U B' N9D2 B N3U' B' N3B2 N9D L N9D' L' N3B2 N3R' L N9D L' N9D' N3R L2 N3F' N8U N3R2 U N8F' U' N3R2 N3F U N3F' N8F N8U' N3F U' L2 N3D2 U2 N8R N3F' N8U' N3F U2 N3F' N8U N3F N3B2 U2 N8R' U2 N3B2 N3D2 N8L2 N3D' N8L' R D2 N8L N3D N8L' N3D' D2 R' N3D N8L' F' N7U' N3B U' N3B' N7U N7L2 N3B U N3B' N3F' U' N7L2 U N3F F B N3R2 B' N7U2 B N3R2 B' N7U2 N7F N3D N7F' D' N7F N3D' N7F' N3B N7R N7F N3B' D N3B D' N7F' N7R' D N3B' N3D L N7D' L' N3F N3D' L N7D2 L' N3D2 L N7D' L' N3D' N3F' N3D' F U' N3R N3B2 U N6R U' N3B2 N6D' N3R' U N3R N6D N6R' N3R' N6U2 R' N3B2 R N6D' N3R N6U' R' N6U N3R' N3B2 N6U' R N6U' N6D F' N6D2 N3L D' N6D' L' N6D N3L' N6D' L N6D' N3L D N3L' L' N6D' L N3F2 L' N6D L N3F2 U2 N3B2 N5R' B N5R N3B2 N5U2 B' N3R' B N5U2 N5R' B' N5R N3R U2 N5U N3L' N5U' L' N5U N3L N5U' L B2 N5D' B2 D' N3L' B2 N5D B2 N3L D N5B N3R' F' N3R N5B' N3R' F N3R SU' N5L N3U D2 N3B2 D2 N5L' D2 N3B2 U D2 N5L N3U' N5L' D' N3D SR N3F L2 N4D' L2 N3F' L2 N4D L' R' N3D' N4U N3U N4R' N3U' N4R B L' N4R' N3U N4R N3U' L B' N4U' N4F2 N3L' F' N4U' F N3L2 F' N4U N3L' N4F2 N3L F N3L' N4F R' N3B2 R N4U' R' N4U N3B2 B2 N3R' B2 N4U' B2 N3R B2 R N4U N4F' N3R' N3D N3R' N3D' N3R N3F' B2 N3R N3F N3R' B2 N3R N3L N3B N3L N3B' R D' N3B' N3L' N3B N3L D R' F2 N3R N3F N3R' N3F' F' N3R F' N3L F N3R' F' N3L D' N3D' N3F' N3R F N3R' F' N3F N3D F N3R F' N3R' D B' U' N3U' N3L U B NL' B' U' N3L' N3U U B N3B NL F' NL' N3B' NL F U' NR U' N3R U VR' NU2 N3R U' N3R' TU2 N3U' L' N3B' N3D2 R ND R' N3D2 N3B R ND' SR' N3U D2 NF' MU2 NF R2 NF' R2 MU2 R2 NF R2 D2 NF2 TD2 F' ND' MR' ND MR F TD2 TR' MF' NR B' NR MF NR' B R NF2 NR2 MF' L2 MF NR2 MF' L2 B2 NU' MF NU B2 D' NB D' N9L2 D NB' D' N9L2 SU2 NR N9D2 N9F2 NR' B2 NR N9F2 NR' B2 U2 NR N9D2 NR' D N9L ND L ND' N9L' ND L' R2 ND N9L ND' N9L' R2 D' N9L ND N9L' ND2 F N9U' F' NU' F NB2 N9U F' N9U' NB2 N9U NU N8D' N8L NL NB' L2 N8U L2 NB L2 N8U' L2 NL' N8L' NR N8D NR' L2 U' NR N8D' NR' N8D U L2 D NL N8F NL' L2 B2 NU L2 NL N8F' NL' L2 NU' B2 L2 D' N7B2 U NF2 U' R' N7B' N7L2 R U NF2 U' R' N7L2 N7B R N7B' B2 L N7B' NR' N7B L' N7B' B2 N7U B2 NR B2 N7U' ND NF NU2 F' N7R F NU2 TF' N7R' F L N7R NF N7R' NF' L' ND' N7L' NF' N7L B' N7L' NF N7L B D' NB2 D N6L NB N6U NB' D' NB N6U' NB D N6L' D N6D' N6B' R' N6D D2 NR' D2 N6D' R N6B N6D D2 NR D L NB N6R' NB' L' NB N6R2 NB' D L2 NB N6R' NB' N6R L2 D' N6R' N6F' N6D L' NF L N6D' N6F L' NF' L N5R' B' NR B N5R B' NR' SF N5L2 F' D' NL' TD F N5L2 F' TD' NL D B2 N5L NL2 D N5F D' NL2 NF2 D N5F' D' NF2 N5L2 NB N5L F' N5L' L' NB' N5L2 NB L NB' N5L' F N4D' NF U2 NF' U2 N4L' U2 NF U2 N4L NF' N4D NL2 N4B' L N4B NL2 N4B' NF' N4R NF' L' NF N4R' NF' N4B L NF2 N4B L' NB' L N4B' L' B' NU' B N4R' B' N4R NU NB2 N4R' B N4R ND2 N4F L' N4F' L ND2 NB' L' N4F L N4F' N3D2 NF' U' NF U N3D2 N3F U' NF' U NF N3F' N3R' NB N3R F N3R' NB' N3R SF' N3L ND NB N3L' B N3L TB' ND' B N3L' SF2 N3R N3B2 F' NR' F NR N3B2 VR' F' NR F' NU2 R' N3B R NU2 R' N3B' R U NB2 ND NB U' NB' ND' NB NR' U NB U NU2 TR' NU NR NU' R TU2 NR ND U NF' ND' F' ND' NF ND F U' ND' NR' B ND NB ND' B' ND NB' ND NR ND' MF2 D U2 F' MU2 F D' U2 F' MU2 F MF2 U F MR F' L' SF SR' B MR' B' SR SF' L U' N9F2 R N9F N9B L2 N9B' N9F' R' N9F N9B L2 N9F F N9B' R U N9B' SU B2 SU' N9B U N9F' D' B2 D N9F U2 R' F' B' N9R2 R' B N9R' B' R B N9R D2 B' N9R2 B N9R2 D2 N9R2 R' D N9L' D' R D N9L D' SR B' R' N9F' R SF' R' N9F R F R2 N8F' R F' D2 R' N8F R N8F' D2 F N8F L N8B L' B' L N8B' L' B L R2 F' N8R F R2 D' L F' D N8R' D' F L' D2 N8L' D L N8D L' SU L N8D' L' U' N8L D' N8L2 B2 R2 N8F2 D' B2 D N8F2 N8L' D' B2 D R2 B2 N8L' N7B R' N7B2 U' N7F D B2 D' N7F' U N7B' SU' B2 SU N7B' R N7B' U' N7R2 U R2 U' N7R2 U R' N7R N7U R' D U R N7U' R' SU' F' N7U F SU2 F' N7U' F U2 N7R F' R2 F N7R2 F' R2 F SU2 L D2 SR N7D2 SR' D2 SR N7D2 R' SU2 B L B' N6R B L' B' N6R' SR U N6R' U' R2 L U N6R U' F N6R2 F' R F N6R2 F' SU L2 R B' N6R' B SR2 B' N6R B R D N6B2 U' F2 U N6B2 U' F2 L2 B' L' N6D L B D2 L' U B' N6D' B U' L D2 R' B' N5R' B L2 R' B' N5R B R2 U F2 U' N5R U' SF' R' SF U N5R' U' SF' R SF U2 F2 U' N5U' R U' R' N5U R U B' U' L' N5U2 L SF D' SF' L' N5U2 L SF D SF' U B N5D L N5D' R' N5D L' N5D' R N4F2 R' D' R D N4F2 N4R2 D' R' D N4R2 N4L B2 U' N4F U B2 N4U2 F' U' F N4U2 N4F' F' U F N4L' L' N4F' L F' L' N4F L B D' R B' N4R' B SR' B' N4R B L' D SF L D L' N3D2 L D' L' N3D2 R2 B' N3R' B R B' N3R SF' N3R' F SR F' N3R F L N3U2 F U' N3F U SF' U' N3F' B' N3U B U B' N3U' D N3F' D' B' R' D R N3F N3U2 R' D' R N3U2 B N3U2 U2 NF2 U F U' NF2 U F' U NF2 U' B' NR' SF R2 SF' NR B' NL F R2 F' NL' B2 U NF2 NB' L B' L' NB L B L2 NB L SF L' NB' L SF' NR' F R' F' NR F R SF' ND B' SU2 B ND' B' SU2 L2 U' B2 U L' F2 L U' B2 U L' F2 L'


""".split()

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
else:
    print(f"Solution is invalid. Diff to WC: {num_difference} > {wildcards}")
    print(f"Expected: {solution_state}")
    print(f"Got: {state}")
    print(f"Writing to partial solution file")

    with open(f"data/sse_partial_sol.txt", "w") as f:
        f.write(".".join(santa_solution))

    cube_scramble = " ".join(list(map(lambda x: move_map[x], santa_solution)))

    print(cube_scramble)

