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
NL' N3D L B R' B' R' L2 B R2 B2 R2 B R L B' R B F2 NR' F' L' F NR F' L ND F' D2 F ND' SF' D' L' NB' L D B' D' L' NB L D' R' B2 NB' U NB U' B2 R U NB' U' NB F' U' NF' NR U L2 U' NR' U L2 F U' NF R2 N3U L N3U' R2 N3U U N3R U' L' U N3R' N3U' D' R F R' L' N3B L R F' R' B' L' N3B' L B N3B2 D F D' N3B2 D N3F2 U' N3F2 U F' L B' L' U' N3F2 U N3F L B L' N3F B N3L' B' R' B N3L B' R D2 F N4U2 F' D2 SF D2 B N4U2 B' D2 SF' N4U L D B N4L' B' D' SR D B N4L B' D' R' N4U' N4B L' B L N4B' L' B' L N4B F N4R' F' L R F N4R F' R' L' N4B' D' R N4D2 R' D R N4D2 R' MF' R' F' R MF R' F R B U B' MU' B U' D' B' MU B D R' MU R B' R' U B MU' B' U' R NL' B' NU' ND' NF U' NF' ND NU NF U NF' B NL TB' NL B' NL' NB NL B2 NR' B' NL' L NR NU NR' NU' L' B NU F NR NB2 NF NU NR' F' NR F NU' TF' NB2 NU' B' N3L N3D' B NU B' N3D B NU2 B' N3L' B NU F' D NR N3U' VR' D' NR D N3R N3U D' NR' F R NB N3L U' NB' N3U NB U NB' N3U' R' NB N3L' N3B NR ND' N3B' R N3D' N3B R' VD NR' N3D' R N3B' N3D R' NB' N4R' D B NL NF2 B' D' N4R D B NF2 NL' B' NR' D' N4F D N4F' NR NU N4F D' N4F' NU' B ND' N4D' N4B' N4L' ND B' ND' B N4L N4B N4D B' ND' N4F' ND' L ND L' N4F N4U2 L ND' L' ND' N4U2 NB MU B' TR' MU NR MU' R B MU' NB' NL' D' R NU' MR' NU R' NU MR D NU' NL MU' L NU L' MU L NU' TL' U' NF MU NF' U NL D' NL' MU' NL D N3F' U' NU2 NB' U N3F U' N3F' NB NU2 N3F U R' NB' N3L VB R NB' R' N3B' R N3L' N3U' NB D' NB' N3U NB NR N3D NR' D NR N3D' NR' B U N3U N3R2 U' B' NL' B U N3R2 N3U' U' B' NL N3B N3L' N3B' L N3B N3U N3L D N3L' N3U' N3L D' N3B' L' N3U' L N3B N3R' N3B' N3R N3B' N3R' N3B L' N3B N3R N3B' N3U R' B' N3R' N3D2 N3R N3D2 B N3R N3F N3R' N3F' R N3B2 N3R N3D' L N3D' N3R2 N3D L' N3D N3R N3B2 N3L' F' N4R N4U F N3L2 F' N4U' N4R' F L' N3L N3D2 L B N4D' B' L' N3D2 N3L2 L B N4D B' N4U F N3R N4U' N4R' N3R' F' N3R F N4R N4U F' N3B' N4B' U N4B U' N3B N3R' U N4B' U' N4B N4U' N4F' B' N4L B N3R' B' N4L' N3R N4F2 N3R' B N3R N4F' N3L MU2 N3L' MU L' MF MU' N3L MU MF' L MU N3L' MF N3R F' N3R MF' N3R' F N3R' U N3B' U' MR' U N3B U' L N3B MR N3B' L' N3F N3L MU' N3L' N3F D N3F N3L' MU N3L N3F' D' N3F2 L NF L' N4D' L N4D NF' NR2 N4D' L' N4D NR2 D' NF' N4U' NF D NF' N4U NF NU2 N4L U' N4L' NU2 N4L' U NL U' N4L2 U NL' N4L D NF' D' N4L2 D NF N4F D' NB' D N4F' D' NB NR N4U NR' U NR N4U' NR' U' N3F' N4R N4D' N3U' F' N3U F N4D N4R' N4F' F' N3U' F N3U N4F' N3F N4L U' N4B N4L' U N4B' N3U N3L N4B U' N4L N4B' U N4L' N3L' N3U' N4F2 N4B' L N3B' L' N4B' L N3B N3U' L' N4B2 L N3U L' N4D' N4F F N4D N4R N4D' N4R' F' N4F' N4D' N4R' N4D N4R R F' N4R' N4D' N4R N4D F R' N4D N4R' N4L2 N4D R N4D' N4L N4U N4L N4U' N4D R' N4D' R' D N4R' B N4R' N4F' N4R B' N4R N4F D' R N4R N4D MF' R N4R MF N4R2 MF' R' MF N4R N4D' L' N4F' MR' N4F MR L' N4U F' MR' N4F' MR N4F F N4U' L2 U' N4L' N4B MU N4B' N4L D' N4B MU' N4B' U D R' N4F' N4U MR' N4U' N4F R' N4F' N4U MR N4U' N4F R2


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

