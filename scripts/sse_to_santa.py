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

U' N4F' R D' L2 D R' D' L2 D U L2 U F' R2 F U' L2 U F' R2 F U2 F ND2 F' U' F ND2 F' U NR' NB' R NB F' D' TB' R' NB R B D F R' NR U' B ND2 B' U B ND2 B R F D F' NR NU2 F D' F' NU2 R' D NR' D' B2 F' NR' D' U' R' ND R U D R' ND' TR F N3F D' F D N3F' D' F' D R2 N3D' D' N3R' D R' L' D' N3R D L R N3D R' B' R' N3F R B R' N3F' N3U F2 D' N3F D F' D' N3F' F' N3U2 F D F' N3U L' U' N3L U L2 R' N3U L' U' L N3U' L' N3L' U R F' D F N3D2 F' D' F N3D2 N4B U' N4B N4R' U' R U N4R F R' N4B' R F' U' R' U2 N4B' N4R' U' L U N4R U' L' U N4D' B N4D' B' R' B D' R N4D R' D B' R N4D F2 D2 B' L' N4B L B D2 F L' D2 N4B' D2 L2 N5B' L' SF L N5B L' B L' U L' N5U' L U' L' N5U L2 N5F F R' N5U' R F' U R' D F N5U F' D' R U' N5F' U N5R D R' N5R' F N5R R D' N5R' N5B2 D R' F' R D' N5B2 U' NU2 R' NF' NR' F' NR' NF NR F R NF' NR U' NF NU2 NF' U NF F' R' NB D NR D' NB' NL D NR' D' NL' R F L NB' NL2 NU2 NL' NU2 B NL' NB NL B' TL' NF NR' NF' L2 NF NR NF' L2 U' L' N3U NL' VU' L N3U L' NU TL N3U' U NB' R NB N3L NB' R' NB VL' N3U' NL' D' NL N3U VL' NU' N3L D N3L' NU N3L2 NL' NF2 N3R U' N3R' U NF2 NL' ND' U' N3R U N3R' ND N3L' N4U N4F2 D NL' NB' D' NB N4F2 N4U' NB' D NB N4U NL D' N4U' L D N4L' NU NF' N4L D' N4L' D NF NU' D' N4L2 L' U' N4L' ND' N4L U N4L' ND NB N4U' NB' D' R' NB N4U NB' N4U' R D N4U N5R' NB N5R F' N5R' NB' N5R' F' U' NR' TU F N5R2 F' TU' NR U F2 N5U' NL N5U R' N5U' NL' N5U R NR' N5U NR U' NR' N5U' NR U R' L' NF' L R N5F' R' N5F' L' NF L N5F R N5F U N3R ND' N3R' U' N3R ND N3R' F' NR F N3F N3D' F' NR' F NR N3D N3F' NR' D NR' N3U F NR NU2 F' N3U2 F NU2 NR' F' NR D' NR' N3U NR U' N3L' U' N3L N3B' N3D N3L' N3B U N3B' N3L N3D' N3B U N3U' N3L2 N3U B' N3L' N3B' N3L B N3L N3B N3F N3L' B2 N3L N3F' N3L' B2 N3L2 B N3L' N3B' F' L' N3B N3L N3B' L N3F' N3L' F N3L N3F N3B' B' N3L' N4R N4D' L' B' N3D B L N4D N4R' L' B' N3D' B L N3L N3B2 N4L' N3U N4L D2 N4L' N3U' N3B' N4L F' N4L' N3B N4L F D2 L' N3B' N4L N3B L N3B' N4L' N3B N3L U N4L N4D' U' N3L' U N3L N4D N4L' N3L' U' N3D N3F N5D2 N5L N3F' R N3F N5L' N3B' R' N5D2 R N3B N3F' R' N5F U' N5F' N3D' N5F N3F U N5L' U' N3F' U N5L N5F' U N5B R' N5B' N3B' N3L' N5B R N3L B N3L' N5B' N3L B' R' N3B R U' N4R' L' N4F L2 ND' L2 N4F' L2 ND L' N4R NF R N4D' NR2 NU N4R' NU' R' NU R N4R N4D R' NU' N4D' NR2 N4D NF' NL' ND N4D N4L' ND' L ND N4L NB2 L' N4D' L NB2 ND' L' NL N3B2 R N4U R' N3B2 N3D R N4U' N3D' N4L N3D N3U' N4L N3U R' N3U' N4L' N3U N3D' N4L' R' B D' N4D' N4R' D B N3R' B' D' N4R N4D D B N3R B2 R N3L' N4F L' N4F' N3L N4F L N4F' B N4L R N4F N4U N4F' N4U' R' N4L' B' N4L2 F N4D' F' N4D N4L2 N4F N4D' F N4D F' N4F2 N4D N4R N4D' N4B L N4D' L' N4B' N4R' L N4D L' N4U D N4R' N4U' L N4U' N4R N4U L' D' N4F L U N4B' N4U' U' L' N5F2 L U N4U N4B U' L' N5F2 L' N5F N4L' N5F' L N5F N4L N5F' N4L B' N4L' B N4L N5U N5L' N4L' B' N4L B N5L N5U' N4L' F R N5R N4F' N5R' N4F R' F' N4F' N5R N4F N5R' N5L' L ND' L' ND N5L N5D ND' L ND L' N5D' NB L NB' N5R NB L' NB' N5R' ND2 F' N5R' ND' N5B' ND F ND' N5B ND' F' N5R NU' NF L' N5U' L NF' NU L' N5U L F L' D N5B2 N5L D' N3F D N5L' N5B2 D' N3F' L N5F N5D N3L' N5D' R' D N5D N3L N5D2 N3L' D' N3L N5D R N5F N3R N5F' L N5F N3R' N5F' L' N5F' L' N3D N3B' L N5U L' N5U' N3B N3D' N5U L N5U' U2 N4R' N5D N4R U2 N4R' N5D' N4R D' N4B' N5U' N5L' N4B D N4B' D' N5L N5U D N4B F N5D N4B' N5D' F' N5D N4B N5D' N4L' F' N5U F N5U' N4L N4F N5U F' N5U' F N4F' R N5U F' N4D F R N5U N5R' R' F' N4D' F R N5R N5U' R' N5U' R' U' N5B' U L N5D2 N5F L' U' N5B U L N5F' N5D2 L' N5U U R' N5F N5R N5F' N5R' R U' N5U' N5B N5D F' N5F' D' N5F N5D' N5F' D N5F N5B2 N5D F N5D' N5B N5L2 N5B' L N5U L' N5L N5F N5L' L N5U' L' N5B N5L N5F' N5L

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
    print(f"Length: {len(cube_scramble.split())}")

