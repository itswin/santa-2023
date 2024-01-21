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

initial_state = puzzle["initial_state"].split(";")
solution_state = puzzle["solution_state"].split(";")

with open(f"data/solutions/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

# with open(args.moves, "r") as fp:
#     moves = fp.read().split()

santa_to_sse = get_santa_to_sse_move_map(n)
print("Santa to SSE", santa_to_sse)

sse_to_santa = get_sse_to_santa_move_map(n)
print("SSE to Santa", sse_to_santa)

scramble = invert(current_solution)
sse_solution = []

move_map = get_inverse_move_map(n, False)
print(move_map)
cube_scramble = " ".join(scramble)
cube_scramble = list(map(lambda x: move_map[x], cube_scramble.split()))
cube_scramble = " ".join(cube_scramble)
print("Cube scramble")
print(cube_scramble)
print()

for move in scramble:
    print(move, "\t", santa_to_sse[move])
    sse_solution.append(santa_to_sse[move])

print("SSE Scramble")
print(" ".join(sse_solution))
print()

# sse_solution = "D R2 L' SF' MU2 SR2 SF2 MD2 SR2 F2 B U2 B' ML D' MF' U' MF D L2 MF' U MF ML' U' L2 U MR' D' MB U' MB' D L2 MB U MB' MR U' L2 U R' L' F2 L R B L' R' F2 L R B'".split()
sse_solution = """
NB N3B2 D NL D' N3B2 D NL' N3F2 D' NB' D N3F2 D' R' ND R N3U R' ND' R N3U' N2U' F ND NL F' N3D2 F NL' F' N3D F ND' NU N3F NU' F' NU N3F' NU' L N3R' B2 N3R NF' N3R' B2 U2 NF' U2 N3R U2 NF U2 NF TL' B N3L B' NL NU2 B N3L B' NU2 B N3L2 B2 NF' N3R B' U N3R' U' NF NR' U N3R U' B N3R' B' NR N3F' N3B' NR B2 N3F L2 N3U' N3F' B2 NR' B2 N3F N3U L2 N3B N3D L2 N3D2 R2 N3L' NB L2 NB' N3L NB SR2 N3D2 R2 NB' SR2 N3D' TR2 ND' R NB' R' NB ND NR2 NB' R NB R B2 NR' ND2 U' NL' U ND2 NR' U' NL U NR2 B2 NF2 D' NR NF2 NR' NF ND2 NF D NF ND2 NF NR' NF' D NB D' NF NL2 D NB' D' TL2 NR SF' SU' B' N3D' B N3D SU F N3D' B' N3D SR' N3F' L F2 B L' N3F L F2 B' R' U N3F' U' R' SF' R U N3F U' R' SF R' D' N3B' D F D' N3B SU' F2 SU N3F SU' F2 SU N3F' D F' N3B L' F L N3B' L' F' L F D' F NU' F' D F NU F' L' NB' L SF' L' NB L B' NU2 R U R' NU2 R U' R' NL' F' ND' F' SR' U SR F ND F' SR' U' SR F2 NL R B R' NF2 R B' R2 U2 NF U2 R U2 R' NF' R U2 NF2 SF U B2 SU B' ND B SU' B' ND' B' U' SF' L D F' U F D' B' F' U' F U B U' L' D2 L' B U2 B' L D2 L' B U2 B' L B2 R' D F2 D' R B2 R' D F2 D' R
""".split()

# 245
# NB N3B2 D NL D' N3B2 D NL' N3F2 D' NB' D N3F2 D' R' ND R N3U R' ND' R M2U' F ND NL F' N3D2 F NL' F' N3D F ND' NU N3F NU' F' NU N3F' NU' L N3R' B2 N3R NF' N3R' B2 U2 NF' U2 N3R U2 NF U2 NF TL' B N3L B' NL NU2 B N3L B' NU2 B N3L2 B2 NF' N3R B' U N3R' U' NF NR' U N3R U' B N3R' B' NR N3F' N3B' NR B2 N3F L2 N3U' N3F' B2 NR' B2 N3F N3U L2 N3B N3D L2 N3D2 R2 N3L' NB L2 NB' N3L NB SR2 N3D2 R2 NB' SR2 N3D' TR2 ND' R NB' R' NB ND NR2 NB' R NB R B2 NR' ND2 U' NL' U ND2 NR' U' NL U NR2 B2 NF2 D' NR NF2 NR' NF ND2 NF D NF ND2 NF NR' NF' D NB D' NF NL2 D NB' D' TL2 NR SF' SU' B' N3D' B N3D SU F N3D' B' N3D SR' N3F' L F2 B L' N3F L F2 B' R' U N3F' U' R' SF' R U N3F U' R' SF R' D' N3B' D F D' N3B SU' F2 SU N3F SU' F2 SU N3F' D F' N3B L' F L N3B' L' F' L F D' F NU' F' D F NU F' L' NB' L SF' L' NB L B' NU2 R U R' NU2 R U' R' NL' F' ND' F' SR' U SR F ND F' SR' U' SR F2 NL R B R' NF2 R B' R2 U2 NF U2 R U2 R' NF' R U2 NF2 SF U B2 SU B' ND B SU' B' ND' B' U' SF' L D F' U F D' B' F' U' F U B U' L' D2 L' B U2 B' L D2 L' B U2 B' L B2 R' D F2 D' R B2 R' D F2 D' R

# move_map = get_move_map(n)
santa_solution = []
for move in sse_solution:
    # print(move, "\t", sse_to_santa[move])
    santa_solution.append(sse_to_santa[move])

santa_solution = ".".join(santa_solution).split(".")
# santa_solution = invert(santa_solution)

print(".".join(santa_solution))

print(f"Validating")
state = np.array(puzzle["initial_state"].split(";"))
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

