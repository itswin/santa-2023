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

U' N4F' N5L' N5U' L' N5U N5L' N5R' N5U' L N5U N5R N5D2 N5U N5L D2 N5L' N5U' N5D2 N5L D2 N5L N5D2 N5U L N5U' N5F L' N5D2 L N5F' N5L N5U L' N5U' N5L' N5U' F' N5U N5B2 N5U' F N5U F' U N5F2 U' N5B2 N5L U N5F2 U' N5L' F U2 N4R' N5D N4R U2 N4R' N5D' N4R2 N5F' D' N5B N4R2 N4U N5B' D N5B N4U' N5F' D' N4R2 D N5B' N5F2 N4R' F' N5R D2 N4B D2 N5R2 D' N4B2 D N5R D' N4B2 D' N4B' D2 F2 N5R' N5D' N4F2 N5D N5R F' N5R' N5D' N4F2 N5D N5R D N5R D' N3L D N5R' D' N3L' N3B2 N5U2 N3B' U' N3B U N5U2 N5F2 U' N3B' U N3B' N5F2 N3F' R N5B' R' N3F R N5B SR' N3U' L N5F N5D' L' N3U L N3U' N5D N5F' N3U N3B N5L2 N3B' L2 N3B N5L2 N3B' N5L' NF N5U' NF D' NF' N5U NF D2 N5L' N5B D' NF2 D N5B' N5L D' N5L D' NF U2 N5R NF U2 N5R' U2 NF' U2 N5R NF' U2 N5R' U2 D NB N5L' NB' L' NB N5L NB' L B' N5R' N4U2 N4F N5R B' N5R' N4F' N5L B N4U2 B' N5L' N5R B2 L N5U N4L N5U' L' N5U N4L' N5U' N5F2 N4R N4U F N4U' N4R' N5F2 N4R N4U F' N4U' N4R' F R N5R N4F' N5R' N4F R' F' N4F' N5R N4F N5R' N4D B2 N4L' B2 N4L L' N4D' N4L N4D L N4D' N4L2 B2 N4L N4R N4U R2 N4F' N4U N4F N4U' N4R R2 N4U' N4R2 B2 N4U F N4U' N4B N4U F' N4U' L2 N4B' N4R2 N4B L2 R N4F2 N4U R' N4B2 R N4U' N4F2 R' N4B N4R2 R N4B N3L' N4B' R' N4B N3L N4B' N4L R N3D R' N4D N4F R N3D' R' N3D N4F' N4D' N3D' N3L U2 N3R U2 N4B2 U2 D' N3R' N4U N3R D N3R' N4U' U2 N4B2 N4L' N3L2 N4F L' N4F' N3L N4F L N4F' N4R' L' N4F L2 ND' L2 N4F' L2 ND L' N4R NF R N4D' NR2 NU N4R' NU' R' NU R N4R N4D R' NU' N4D' NR2 N4D NF' NL' ND N4D N4L' ND' L ND N4L NB2 L' N4D' L NB2 ND' L' NL N3F N5L' R2 N5R' N3B' L2 N3B N5R N3B' SR2 N5D2 R2 N3B R2 N5D2 N5L F N3L' F N5L F' N3L F N5L2 F2 N3D' F2 N5L F2 N3D N3F2 N5R' U' N5R U N3U N3L U' N5R' U N5R N3L' N3U' N3F N4U N3D N4B N3D' N4D' N3B N4D F N4D' N3B' N4D N3D N4B' N3D' F' N4U' N4L' N3U N4L D2 N4L' N3U' N4L D2 N3L U N4L N4D' U' N3L' U N3L N4D N4L' N3L' U' R' L' N4D N4U' L N3B2 L' N4U N3L2 N4D' L N4D N3L2 N3B2 N4D' N3F R N3B2 R' N3F' R N3L' F N3F N3L2 N3U N3L2 N3U' F' N3L F N3L' N3F' N3L N3B2 F N3R' B2 N3R' N3B' N3R N3B B2 N3R F2 B N3L' N3B' F' L' N3B N3L N3B' L N3F' N3L' F N3L N3F N3B B' N3L N3U' NF NR2 F2 N3U F2 NR2 NB' F2 N3U' F2 NF' N3U NB N3L' D2 N3L N3F' D' NB' D NB N3F N3L' NB' D' NB SU N3R ND' N3R' U' N3R ND N3R' N5D' L NU' L' N5D L NU L' NF2 N5R B F2 NU2 F2 N5R' F2 NU2 TF2 N5R B' N5R' D' NB N5L2 N5B' NB' R' NB NU2 R N5B R' NU2 N5L2 NB' R D L2 NU2 L' N5F' L NU2 L' N5F L2 N4B' NU2 NB N4U' NB' D' NB N4U D N4B D' N4B' NB' NU2 N4B D N4U' L N4U NL2 N4U' L2 NB2 L N4U L' NB2 NL2 SR2 ND N4F U2 N4F' ND' N4F U2 R2 ND2 R2 N4F' R2 ND2 NL F2 NU' N3F NU F2 NU' N3F' NU NL' D' N3L' NU' N3L NB D N3L2 D' NB' D N3L NU N3L R2 NU' N3R NU N3R' R2 B2 N3R NU' N3R' NU B2 U2 N3F NU N3F' U' NB' U' N3F U NB TU' N3F' L' NU' L N3U L' NU L N3U' NB' NL NB ND' B' ND NB' NL' NB ND' NF' NL B NL' NF ND F' NL' F NR2 ND2 F' NL F NL' ND2 TR2 NB' R' B NR' NB' NR B' NB2 NL NB' R NB R' U2 NB U2 NB' NR2 NB U2 NB' U2 R' NR2 F R2 F' N5R2 F R2 F' N5R2 F' L N5B' L' SF L N5B L' B' U N5L2 F SU R' SU' F' N5L2 F SU R SU' F' U' B2 N5B2 U2 F' N5U' F SU' F' N5U F D' U' N5B2 F2 D2 B' L' N4B L B D2 F L' D2 N4B' D2 L F D B' N4D B D' B' N4D' U B' R N4U R' SU R N4U' R' SU' B U' B R' SU N4L2 B' R2 B N4L2 F' N4R' SF R2 SF' N4R F SU' R B N3D2 B' D' B N3D2 B2 N3R SU' B' R' N3U' R B SU B' R' N3U R B N3R' B D B' L' B' N3L' B L R B' N3L B R' B R SF R U2 N3B' U2 R' SF' U2 R N3B R' U2 SR' U L' N3U2 L U' L' N3U2 R' ND R ND' U' B' ND R' ND' R B U F' NL2 D' L ND2 R' NU SR D2 SR' NU' R ND2 L' D' NL2 NU2 F D F' NU2 F D' B SU NB' D NB U' TB' D' NB' D F D' NB2 D ND2 F' U' F ND2 F' U R D' L2 D R' D' L2 D U L2 U F' R2 F U' L2 U F' R2 F U2

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

