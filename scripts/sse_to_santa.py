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

with open(f"data/sse_scramble.txt", "w") as f:
    f.write(" ".join(sse_scramble))

# sse_scramble = "D R2 L' SF' MU2 SR2 SF2 MD2 SR2 F2 B U2 B' ML D' MF' U' MF D L2 MF' U MF ML' U' L2 U MR' D' MB U' MB' D L2 MB U MB' MR U' L2 U R' L' F2 L R B L' R' F2 L R B'".split()
sse_scramble = """
R D SF2 U NL' N3L N4D N5B' R2 F' N5L N5B N5L2 N5B N5L F N5L' N5B2 R2 N5L' U' N5B' U N5B N5L2 N5F2 N5D2 N5B' U' N5B U N5D2 N5F2 N5D2 N5R' N5F' N5D N5F L' N5F' N5D' N5F N5R N5D L N5D2 L2 N5B' B' R N5R' N5B' N5R N5B R' B N5B L2 N5D' SU B N5R B' D' N4B N4L D B N5R' B' D' N4L' N4B' U D2 N5L' N5U' U' N4R' U N4R N5U N5L N4R' U' N4R U' N5L2 N4F' L2 N4F N5L' L' N5F' L N4F' L' N5F L' N4F L2 N5F L2 N4D' L2 N5F2 N4D N5F D' N5F' N4D' N5F D L2 N4D N5L' F' N5U' N3F N5U F N5U' N3F' N5U N5D B N5D' N3B' N5D B' N5D' N3B' N5R' F' N5R F N3B2 N3R F' N5R' F N5R N3R' N3U2 N5B2 L' N3F L N5B2 N5U2 L' N3F' L N5U2 N3U2 F N3L B2 U2 N5F U2 N3L' U2 N5F' U2 N5F' N3L B2 N3L' N5F F' N5U2 NR B' N5R B NR' NU2 B' N5R' B NU2 N5U2 NL2 U2 N5U N5F' U2 B' NL B U2 N5F N5U' U2 B' NL' B NL2 D' NR N5D NR' D F NR N5D' NR' N5D ND F' N5D F ND' F' N5D2 F' N4D R N4D' N5D N5R N4D R' N4D' N5R' N4B' R N5D' R' F N4B2 N5D' N4B' N5L U N4R U' N5L' U N4R' U N4B N5D N4B' N4L N5U R N5U' N4L2 N5U R' N4L U2 N4L' N5U' N4L N4F' D' N4F N5U' N5D B' N5D' N4F' N5D B N5D' D N4F N5U N4L2 F' N4U' F N4L2 F' D' L R N4U' N4L N4U L' R' N4U N4L' D F D' N4F' L2 N4F N4L' N4F' L2 F2 N4U' N4F2 N4U F2 N4F' N4L F N4R' N4F' N4R' N4F N4R2 F' D R2 N4F N4R N4F' R2 N4F F N4R' N4F' N4R F' N4R' N3D' N3B' N4D' N3B U2 N3B' N4D N3B N4D' N4L U2 N3B' U2 N4L' U2 N3B N4D N3D N4F N3F L B N4U' B' L' N3F' N3D2 L B N4U B' L' N3D2 N4F' N4L N3U2 R' F' N4U N4F F R N3U2 R' F' N4F' N4U' F N4L' L N4B L' N3F SR' N3F R N4B' R' N3F' SR N3F' R NL ND' B' N4L B ND NL2 B' N4L' B NL ND N4D NR N4D' L' N4D NR' N4D' L ND' D NB2 U' L' N4B' L U NB2 NU U' L' N4B L NU' SU F' NR' F N4U2 F' NR ND F N4U' F' ND' N4U' NF N4U F N4U' NF' N3R2 L U' N3B2 N3F U N3F' N5R2 N5U N3F U' N3F' N5U' N3B2 U L' N5R2 N3R2 N5L2 N3U F2 N3D F2 N5L' F2 N3D' F2 N5L N3U' N5L2 R' N5B R N3F2 R' N5B' R N3F2 N3D F N3D F' L2 N5U2 N5B L2 F N3D' F' L2 N5B' N5U2 L2 N3D' N3F N3R' N4D B N4D' B' N3R N3F' B N4D B' N4D' N3U' L N4U2 L' N3U2 L N4U2 N3U2 L' N4D N4F2 L N3U2 L' N4F2 N4D' N3U' L' N3B' L N4D' L' N4D N3B N3L N4D' L N4D N3L' N4U2 N4B' D' N4B N3D N3B' N4B' D N4B N4L D' N3B D N4L' N3D' N4U2 N3L2 B U2 N3R N3D' N3R' U2 N3D N3R' N3B N3R N3B' B' R U N3L' N3B2 N3L N3B2 U' N3L N3U N3L' N3U' R' N3L2 L2 N3R' B' N3R N3F' N3B' N3R' B N3R N3B N3F L N3R' D U N3R' N3U' N3R D' U' N3R VU L NU' N3F N3R' NU L' NU' L N3R N3F' NB U NB' N3D2 NB U' NB' N3D2 R N3L' N3U2 R' B' ND' B R N3U2 N3L' R' B' ND B N3L2 NU' N3L N3R D' N3R' NF2 ND' N3R D N3R' ND N3L' D' NF2 D2 NU N5B' D' NB D N5B SU NB' N5U2 NB N5R U' NB2 U N5R' U' NB N5U2 NF' F N5U' F' NL2 NU F N5U F' NU' NL2 NF N5B2 NR' F' N5B L2 N5B' NR N5B L2 N5B NR' F NR SF2 NU' B2 L N5L N5F' L' B2 NU B2 L N5F N5L' L' F2 N4L ND' N4L' D2 N4L ND D' NL' D N4L' D' NL D' NR2 B2 N4R' L2 NF' L2 N4R B2 N4D N4R' L2 NF L2 N4R N4D' NR' B' NL' N4U' NL B NR2 NL' B' N4U B NL NR U N4B NU2 N4B' ND U2 N4B TU2 NL2 U2 N4B' U2 NL2 U' R N4D' R' ND' R N4D R' NF' NU' B2 N3F' U2 N3L' N3F B2 NU B2 N3F' N3L U2 B2 VF D NL N3D2 N3F' NL' U' NL NR U N3F U' NR' N3D2 NL' SU N3L ND N3L' D N3L ND' N3L' NB N3U' NB' D' NB N3U NB' D2 TR NB' NR' NB R' D2 ND U NR' U NB NR NB' ND' NR' U' NR U' ND NL2 ND' L' ND NL2 NR' ND2 NR ND L B' L NF NL NF' R' L' NF' NL' NF R B N5D L' R2 D' N5R' D L R D' N5R D R N5D' B' U' F N5L2 SU' R SU N5L2 SU' R' SU F' U N5R' D2 B' N5D B SU2 B' N5D' B U2 N5R N5B' N5U2 B L F' U' SF L' N5U2 L SF' U F L' N5B U' N5F U B2 U' N5F' U B2 N4U F' U' F N4U' F' U F N4F SR2 B N4R B' R L2 B N4R' B' R N4F' L N4B' D' F' N4D F SU2 F' N4D' F D' U2 N4B R' F' U N4B U' SF2 U N4B' U' F' B2 SR N4B2 L' F L N4B2 L' F' L' B N4L SF N4L2 F' L2 F N4L2 SF' N4L' B' L' N3D SR' N3F' R N3F U2 N3F' R' N3F SR N3D' L U2 R2 U N3R2 U' R2 U N3R2 SU' R' D SR B N3U' B' SR' D' SR B N3U B' L D2 N3D' L' D N3L N3F D' N3F' L N3F D N3F' N3L' N3D F2 R D N3F D' R' F2 R D N3F' D' R' ND' L2 ND L' SF' U' SF L ND' L' SF' U SF R' B NR B' R2 L' B NR' B' R' ND TR F SR' F' NR' F SR F' R2 NB R' B' R NB' R' B2 U2 NL' U2 B' R' U2 B NL B2 ND B U2 B' ND' B R' U F2 U2 R U' B U' F2 U2 F2 U' B' U R' F2 U F D2 F R' U2 R F' D2 F R' U2 R F2

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

