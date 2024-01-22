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
print(cube_scramble)
print()

sse_scramble = []
for move in scramble:
    # print(move, "\t", santa_to_sse[move])
    sse_scramble.append(santa_to_sse[move])
print("\nSSE Scramble")
print(" ".join(sse_scramble))
print()

with open(f"data/sse_scramble.txt", "w") as f:
    f.write(" ".join(sse_scramble))

sse_scramble = """
D' F' R U N5U N5B' N5U' B' R2 N5U N5B N5U' N5B' R2 B N5B R' D N5L2 N5D' D N5L N5D N5L' D2 N5R' D N5L2 D' R N5R' F N5R' N5B N5R F N5R N5B' N5R' F2 N5R2 L' N5B L N5L N5F' L' N5B' L N5B N5F N5L' N5B' N5R' R' N5U' F' N4D F N5U SF' N5U B' N4D' B N5U' SF N5R B' N5B' N4R N5B N4R' B R N4R N5B' N4R' N5B N5R2 N5D N5R' N4R' D N4F' N4R D' N4F N5R N5D' N4F' D N4R' N4F D' N4R N5R2 N5L2 D F N4L N4U F' D' N5L2 D F N4U' N4L' F' D' N3U2 L N5B L2 N3U' L2 N5B' L2 N3U L N3U2 U N3L2 U' N5L2 U N3L2 U' N5L N3D' F L' N3D N5L N3D' N5L' L F' N5L N3D B2 N3L' N5F N3L B2 N3L' N5F' N3L N5U2 D2 N3B' D2 N5R' N3B D2 N5R2 D2 N3B' D2 N5R2 N3B D2 N5R N5U2 NF NL2 NF U' NF' N5F N5D2 NF U NF' N5D2 NL2 U' N5F' U NF' ND2 U N5F' U' N5F ND2 NR2 N5F' U N5F U' NR2 B N5L' B' ND2 B N5L NL N5B' NL' B' NL N5B NL' ND2 N5U' R NU' R' NU N5U N5R2 NU' R NU R' N5R2 U2 N4B' N5U2 N4B U2 N4B' N5U2 U' N5F D' N5F' N4U' N5B D N4R2 D' N5B' N5F D N5F' N4R2 N4U U N4B F' N4U N5B' N4U' N5B N4B F D N5B' N4U N5B N4U' D' N4B' N5R' B N4L' B' N5R B N4L B' N4R2 B' N4U' B N4R2 B' N4U B' N4U N4L' U' N4L' N4D N4L2 N4D' N4U' N4L' U N4L B2 N4D' N4F L2 N4U' N4L' N4U2 N4L N4U' L2 N4F' N4D L' N4L' D' N4L N4U N4F' D' N4R2 D N4F N4L' D' N4R2 D2 N4L N4U' L N3F R N4D' R' B' N3U B R N4D N4R2 R' B' N3U' B N4R2 N3F' N3U' N4B' N3U F N3U' N4B N3U F' N3R N4D N3R' D2 N3R N4D' N3R' D N4F' D N3R2 D' N4F N3R N4D' N3R' D N3R N4D N4U N3R' U' N3R N4U' N3R' U N3R2 NB U NR U' N4R' N4B2 U NR' U' N4B2 N4R NL' D' N4L' D N4L NL NB' N4L' D' N4L D NF N4F N4U2 NF' R NF R' N4U2 N4F' R NF' N4U R' NB2 R N4U' R' NB2 L NF N4L' NF' L' NF N4L L2 NF' ND' N4R' ND NF L2 NF' ND' N4R ND N3U' R' N5B2 N3L2 N5F R N5F' N3L2 N3U2 N5F R' N5F' N5B2 R F' N3U' N5F N3U F N3U' N3L F N3L' N5F' N3L F' N3L' SU N5L' D N3R D' N5L D N5D2 N3R' U2 N3R N5D2 N3R' U N3F2 D L N3F' L' N5B' N5D2 L N3F L' N5D2 N5B D' N3F2 N3B' U N4B' D N3R D' N4B N4L2 D N3R' D' N4L2 U' N3B U' N3L2 U N4B N3D' N4B U' N4B' N3D N4B N3L2 U N4B2 SR N4D R N3U' R' N4D' N4B R N3U R' N4B' SR' N4U N4F R' N3D' R F2 N3R' N4F' N3R F2 N3R' R' N4F N3R N4F' N3D R N4U' N3F' N3R N3B N3R' N3B' L2 N3U L2 N3B N3R N3B' N3R' L2 N3U' L2 N3F R' N3F' N3U B' N3D B N3U' N3L B' N3L' N3D' N3F N3L B N3L' R U N3L' U N3F' N3L L2 N3U L2 N3L' N3F U' N3L L2 N3U' L2 U' ND2 R N3B' NB R N3F2 R' NB' R N3F2 R2 ND2 R N3B R' NB' N3R B N3R' NB N3R B2 NL NF2 B N3R' B' N3R NF2 NL' N3R' B N3B' R' N3B NL' N3B' R N3B N3F2 NL' F' NL N3F2 NL' N3D2 F NL2 F' N3D2 F ND U N5L2 F' ND' NL F NL' N5L2 N5F NL F' NL' N5F' ND F U' N5U ND' NL N5U' L' N5U NL' N5U' SR' NF2 R N5D' R' N5D NF2 NR N5D' R N5D NR' N5F2 U NB U' N5F2 U NB' SU' N5B D' NB2 D N5B' NB' N5D NB D' NB' N5D' NB' NR N4U U' NR U2 N4F U2 NR' U2 N4F' U' N4U' NR' N4F2 D NL2 N4F ND N4F' D' N4F ND' N4F D NL2 D' NU2 F' N4R' F NU2 F' N4R F' NL N4B NL' F2 D NL N4B' NL' N4B D' N4B' L' N3U' L ND' L' N3U L' ND N3R ND' L2 ND N3R' NR2 D' N3B D NR2 D' N3B' NF D L' NF' L N3F L' NF L D' NF' D N3L' TU NR U' B' N3L2 B U NR' TU' B' N3L2 B N3L N3F' NR B' D' NB NU NB' D B NR' B' NU' B D' NR' NU NF' NU2 NR D' NR' NU2 NF NU' NR NF' D NF D' ND2 TF NU F' ND2 F NU' F' R' ND2 NF' ND2 NF R NF' D2 F' R N5B' R SF' R' N5B R SF R2 F N5F D2 N5U' N5B' D B' D' B N5B N5U B' D B D N5F' N5R2 L U' N5R' U SR U' N5R U R' N5R2 D' N5F2 D B D' N5F2 SU' N5B2 U B' U' N5B2 R' F' N5R F R2 L F' N5R' F R' L' SU R' B2 U' N4R2 U F SR F' U' N4R2 U F SR' F' B2 D N4L' D' R D N4L N4U' L' U B SU' N4B2 SU B' SU' N4B2 D' L N4U N4R D' N4R' U' N4R D N4R' U N4D' B SU B' R N4U' R' B SU' B' U R N4U R' U' N4D D2 R' N3D R D2 R' N3D' N3U' R D N3B SU F2 SU' N3B' D' N3F2 U F2 U' N3F2 R' N3U R U L N3U2 N3B' L' U2 SR' U2 R N3B N3U2 R' U2 SR U L' N3B D F2 N3D2 B' D' B N3D2 N3B' B' D B F2 D' F' N3U' F U F' N3U F U' L B2 NU' R2 D2 B' R2 B NU NR2 B' R2 B NR2 D2 R2 B2 U2 R' ND R U2 R' ND' R SF2 L NB L' F2 B' L NB' L' B' D2 L NU2 L' D2 L NU2 F' L' NF2 NB2 L F2 L' NB2 L F' L' TF2 D' B U2 B2 U2 D R2 U2 D' B D F' U2 F2 R2 F
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

