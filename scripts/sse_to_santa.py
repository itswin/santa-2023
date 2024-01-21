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

initial_state = np.array(puzzle["initial_state"].split(";"))
solution_state = np.array(puzzle["solution_state"].split(";"))

with open(f"data/solutions/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

# with open(args.moves, "r") as fp:
#     moves = fp.read().split()

santa_to_sse = get_santa_to_sse_move_map(n)
print("Santa to SSE")
print(santa_to_sse)
# for k, v in santa_to_sse.items():
#     print(k, "\t", v)

sse_to_santa = get_sse_to_santa_move_map(n)
print("SSE to Santa")
print(sse_to_santa)
# for k, v in sse_to_santa.items():
#     print(k, "\t", v)

scramble = invert(current_solution)

state = solution_state
for move_name in scramble:
    state = state[moves[move_name]]

# assert np.all(state == initial_state)

# if n % 2 == 0:
#     center_orienting_seq = []
# else:
#     initial_state, center_orienting_seq = orient_centers(initial_state, moves, n, solution_state)
#     print("Center orienting seq")
#     print(center_orienting_seq)
#     scramble += center_orienting_seq

move_map = get_inverse_move_map(n, False)
print(move_map)
cube_scramble = " ".join(scramble)
cube_scramble = list(map(lambda x: move_map[x], cube_scramble.split()))
cube_scramble = " ".join(cube_scramble)
print("Cube scramble")
print(cube_scramble)
print()

sse_scramble = []
for move in scramble:
    # print(move, "\t", santa_to_sse[move])
    sse_scramble.append(santa_to_sse[move])
print("SSE Scramble")
print(" ".join(sse_scramble))
print()

# sse_scramble = "D R2 L' SF' MU2 SR2 SF2 MD2 SR2 F2 B U2 B' ML D' MF' U' MF D L2 MF' U MF ML' U' L2 U MR' D' MB U' MB' D L2 MB U MB' MR U' L2 U R' L' F2 L R B L' R' F2 L R B'".split()
sse_scramble = """

D' ND F SR F L F' SR F L F' R2 L' F L F2 L' F2 B L' NF L SF2 L' NF' L B F NR2 F' L F NR2 F' L B D' NB' D SF' D' NB D B2 F L2 D2 L ND2 L' D2 L ND2 NL B' D NF2 D' SF2 D NF2 D' B' F2 NL' N3U F2 R' SF U N3L' U' SF' R SF U N3L U' B F M2U' L' R' U' N3L' U R L U' N3L U N3D' D2 R B' R' N3B N3L' R B R' D2 B' D2 N3L N3B' D2 B R U' F N3L F' SR' F N3L' F' SR N3L U R' U' N3L' U F' ND' NB' ND NB F' D' NB' ND' NB TD F' NL F' NU' F NL' F' NU NB U' NB' ND NB U R NU' NL2 NU' R' NU NL2 NU' NB' ND' R NU2 NF' TR NB' TR' NF R B2 NR NB NR' B2 R2 TD2 NF' D NL' D' NF TD R' ND' NL ND R TD B' N3D NB N3D' TB' F' ND F N3D F' VD' NB N3D F NB' N3D' B2 D' ND2 N3F' NR D2 N3B D2 NR' D2 M2F ND2 N3F' D2 N3F D R' NF2 R N3F2 R' NF2 R D' R' ND' R2 N3F R2 ND R2 N3F' R' D N3F2 U NL' N3U' VL U' NL2 U N3L' U' NL N3U NL ND B N3U B' ND' B N3U' B' N3U NR' ND2 N3B' L' M2F' L ND2 L' N3F NR N3B' L N3B N3U' R N3L NU' N3L' NU R' D2 NU' N3L NU N3L' D2 N3R F NR' F' N3R' F NR F' N3R2 R2 N3D R2 NF R2 N3D2 R' NU2 R N3D R' NU2 R' NF' N3R' D' N3R N3D2 N3R2 N3D2 N3R D N3R' N3D2 N3R2 N3D2 N3R' R2 B N3R' M2F N3R2 N3F' N3R' B' N3R' N3B R2 U' N3F U N3B2 N3U' U' N3F' U N3F N3U M2F' D' N3L D N3B D' N3L' D B N3D2 B' N3U' B2 N3D2 N3R' N3D2 N3R B' N3R' N3D2 N3R B' N3U
""".split()


# CR CF2
# B D' L' B R U2 SF2 D2 L B' R' U2 SF2 D' B R' D F2 D' R B2 R' D F2 D' R
# B NL' B L B' NL B L' B' D' R' SU' F NL' F' SU R' SU' F NL F' SU R2 D L U' NL' U SR U' NL U R' B' U' F2 U NF2 U' F2 U NF2 L2 ND2 R ND2 R' D' B' D' R ND2 R' ND2 D B SU' NR U L2 U' NR' U
# B' MU2 B' MU2 B L D' L2 B' MU2 B MU2 L2 D L' B MR' D' R2 SU' MR' D' MR U R2 MR' D MR2
# NU' NB' U NF' NU B NU' NF NB NU NB' B' U' NB F' ND2 R ND' NF2 ND NF2 R' NF2 ND' NF2 ND' F NU' F' NR' NF2 NB' NR F NR' NB NF2 NR NU NB' R' NF' R NB R' NF R NF ND NF' ND U NR' ND' NR U' ND'
# MR' L NF' MR F2 MR NF MR' F2 L' NB2 R U2 B' MU' NB MU NB' B U2 R' NB2 MF' L MF NR2 MF' L' MF NR2 NL2 D NL2 MF2 NL' MF2 NL D' MR' D NL2 D' MR D2 NR D2 L' MF L D2 NR' ND2 D2 L' MF' L ND2 NL2


# move_map = get_move_map(n)
santa_solution = []
# santa_solution = center_orienting_seq
for move in sse_scramble:
    # print(move, "\t", sse_to_santa[move])
    santa_solution.append(sse_to_santa[move])

santa_solution = ".".join(santa_solution).split(".")
# santa_solution = invert(santa_solution)

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

