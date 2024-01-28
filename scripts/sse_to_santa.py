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
NR N4D' U L' SU F' U2 R' U2 R F D R' U' L U' B D2 B' R U2 R' B D2 B' R U R2 B' ND' B ND R2 F U2 ND' B' ND B U2 F' U NB R' U' L' NU' L TU R NU' L' NU L NB' R NF L U F' U' L' NF' L F ND F' U F ND' U' L' R' U B' ND B U' B' ND' SF' R U' N3L2 N3D' U R' U' R N3D N3L2 R' U F U L N3U' N3B L' B' L N3B' N3U L' N3U' B N3U U B R' B' N3D B R B' U2 R' N3D' F N3U2 F' D' F D N3U2 N3F' D' F' D N3F R D L' D' L N4L N4D' L' D L D' N4D N4L' D L' N4L' D2 R N4L F' N4L' F R' D2 F' N4L F L D' F' N4U N4R' F U' F' U N4R N4U' U' F R' N4U' R U R' N4U R N4B L' B2 F D' L' N4F' L D F' B2 L' D' N4F D L2 N4B' L' NB NR NB' NR' L B' NR NB NR' NB' B NF ND2 NU NL' NB' D NB NL NU' ND2 NB' NL' D' NL NB R' NF ND NF' NL' ND' R ND NL U NF ND' NF' U' NR D' NR' NU NF NU' NF' NR D NR' NF' B U' N3R' U F NB NL F' U' N3R U F NL' TB' F' L N3F NL' N3F' L2 NF' L N3F L' NF TL N3F' N3L' NR' NU F NR N3U N3B NR' F' NU' NR F NU N3B' VU' F' NB' R' TB N3L NB' N3L' B' N3L R NB N4B' NF2 N4D R N4D' R' NF2 NL R N4D R' N4D' NL' N4B NR' N4R' NU' N4R NU B D' NU' N4R' NU N4R D B' NR F' TD F ND' N4F F' N4U F ND F' N4U' N4F' D' N4F ND' N4F' F NB N3R NB' R2 NB N3R' VB' NL' N3B L N3B' NL N3B L' R2 NB2 VL N3B NL' B NL N3B' B' N3L' B NL' B' D N3R' D' NB2 D N3R N3L' N3B2 U' NR U N3B2 NU N3L U' N3L' NU' NR' N3L D' N3F ND' N3F' U N3F ND N3F' N3U2 L N3D2 N3B' U N3B' N3D2 N3B U' N3B L' N3U N3B2 N3L B' L' N3L' N3B N3L N3B' L B N3B N3L' N3U F' N3U' N3B N3U2 N3D N3F N3L' U' N3L U N3F' N3U' U' N3D' N3L' U N3L F N4L' F' N3L' N4F' N3L F N3L' N4F N3L N4L N3B' N4D F N4D' N4R' B N4R N3B N4R' B' N4R N4D F' N4D' N4U N3F U N3F' N4U' N3F U' N3F' U2 N4B' N3D' N4B U2 N4B' N3D N3L B' N3L' N4B N3L B N3L' NF' D N4U N4R' D' NF D NF' N4R N4U' NF D' L NF' L' NF N4F' N4L' NF' L N4L F N4L' NF N4L F' L' N4F NB N4U' F N4U NB' N4U' F' N4U B NL N4B' NL' B' NL N4B NL' N4B D' N4B' D L NU NB' L' D' N4B D L NB NU' L' N4B' U' N4D N4F2 U N3R U' N3R' N4F2 N4D' N3R U N3R' D N3B U N4F' N4L2 U' N3B' U N4L2 N4F U' D' R' N4B F' R' N4R' N4D' R F N3D' F' R' N4D N4R R F N3D N4B' R B' N4U' B N3U B' N4U B N3U' U N4F' N4L2 B' N4F' N4D N4F N4D N4L N4D2 N4L' B N4L2 N4F N4B2 N4U' F N4U' N4B' N4U F' N4U N4B' U' N4U B N4U' N4F2 N4U B' N4U' R2 N4U' N4L' N4B N4U R2 N4U' R2 N4B' N4L R2 N4U N4F2

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

