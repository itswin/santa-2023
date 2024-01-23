#!/usr/bin/env python3
import argparse
import pandas as pd
import itertools
import numpy as np
import subprocess
from util import *

# seq 210 234 | xargs -P 4 -I {} python3 scripts/solve_reskin.py {}

def reskin(state, edges, edge_map, odd_centers, odd_center_reskin_map):
    new_state = state.copy()

    for edge in edges:
        print(edge, state[edge[0]] + state[edge[1]])
        new_state[edge[0]] = edge_map[state[edge[0]] + state[edge[1]]][0]
        new_state[edge[1]] = edge_map[state[edge[0]] + state[edge[1]]][1]
        print(new_state[edge[0]] + new_state[edge[1]])
    
    for odd_center in odd_centers:
        new_state[odd_center] = odd_center_reskin_map[state[odd_center]]

    return new_state

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("--sol_dir", type=str, default="data/solutions")
parser.add_argument("--add_picture_state", action="store_true", default=False)
parser.add_argument("--partial_sol", type=str, default=None)
parser.add_argument("--solve_sub3", action="store_true", default=False)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
print(puzzle)

puzzle_type = puzzle["puzzle_type"]
n = int(puzzle_type.split("/")[-1])
moves = get_moves(puzzle["puzzle_type"])
print(f"Number of moves: {len(moves)}")

initial_state = puzzle["initial_state"].split(";")
solution_state = puzzle["solution_state"].split(";")

STICKER_MAP = {
    'A': 'U',
    'B': 'F',
    'C': 'R',
    'D': 'B',
    'E': 'L',
    'F': 'D',
}

state = np.array(initial_state)
solution_state = np.array(solution_state)
print(state)
print(solution_state)

if solution_state[0] == "N0":
    m = {}
    faces = "ABCDEF"
    for j in range(6):
        for i in range(n ** 2):
            m["N" + str(j * n ** 2 + i)] = faces[j]

    state = np.array([m[s] for s in state])

print("INITIAL", state)

move_map = get_move_map(n)
# print(move_map)

if args.partial_sol:
    with open(args.partial_sol, "r") as fp:
        sol = fp.read()
        delimiter = "." if "." in sol else " "
        sol = sol.split(delimiter)

        center_orienting_seq = []
        if sol[0].lower() != sol[0]:
            for move in sol:
                center_orienting_seq.extend(move_map[move].split("."))
        else:
            center_orienting_seq = sol
    for move_name in center_orienting_seq:
        state = state[moves[move_name]]
    print("PARTIAL", state)
else:
    if n % 2 == 0:
        center_orienting_seq = []
    else:
        state, center_orienting_seq = orient_centers(state, moves, n)

    print("ORIENTED", state)
    print("ORIENT_CENTERS", center_orienting_seq)

pre_reskin = np.array(initial_state)
for move_name in center_orienting_seq:
    pre_reskin = pre_reskin[moves[move_name]]
print("PRE RESKIN FACES after orienting centers")
print_faces(state_to_faces(pre_reskin, n), n)

n2 = n ** 2
NORMAL_SOLUTION = "A" * n2 + "B" * n2 + "C" * n2 + "D" * n2 + "E" * n2 + "F" * n2

solution_faces = state_to_faces(solution_state, n)
normal_solution_faces = state_to_faces(NORMAL_SOLUTION, n)

print("SOLUTION FACES")
print_faces(solution_faces, n)

print("NORMAL SOLUTION FACES")
print_faces(normal_solution_faces, n)

edges = get_edges(n, skip=2)
odd_centers = get_diff_odd_centers(n)

print(f"Edges {len(edges)}", edges)
print(odd_centers)

edge_map = make_edge_reskin_map(edges, solution_state, NORMAL_SOLUTION)
print(edge_map)

odd_center_map = make_odd_center_reskin_map(odd_centers, solution_state, NORMAL_SOLUTION)
print(odd_center_map)

# Make sure no edge indexes are in the odd_centers
assert len(set(odd_centers) & set(itertools.chain(*edges))) == 0

print(state)
faces = state_to_faces(state, n)
print("INITIAL FACES")
print_faces(faces, n)

state = reskin(state, edges, edge_map, odd_centers, odd_center_map)
print("RESKINNED", state)
print(type(state))

state = "".join(STICKER_MAP[c] for c in state)
faces = state_to_faces(state, n)

if args.solve_sub3:
    faces = get_3x3_faces(faces, n)
    print("INITIAL FACES")
    print_faces(faces, 3)
else:
    print("INITIAL FACES")
    print_faces(faces, n)

print(faces)
cubestring = make_cubestring(faces)
print(cubestring)

SOLVER_PATH = "/Users/Win33/Documents/Programming/rubiks-cube-NxNxN-solver/rubiks-cube-solver.py"
cmd = [SOLVER_PATH, "--state", cubestring]
# cmd = ["cat", "scripts/split.py"]

if args.add_picture_state:
    cmd.extend(["--picture_state", ";".join(pre_reskin).replace("N", "")])

out = subprocess.check_output(cmd)

out = out.decode("utf-8").strip()
out = out.split("\n")

# Search for the solution line
for line in out:
    if line.startswith("Solution: "):
        sol = line.split(":")[1].strip()
        break

# sol = "D' B  L' U' D2 F' L  U  L  U2 F' L' D2 L  F2 R' L' D2 F2"
print(sol)

# Map it back to our move set
mapped_sol = []
for move in sol.split():
    mapped_sol.append(move_map[move])

solution = center_orienting_seq + ".".join(mapped_sol).split(".")

mapped_sol = (".".join(center_orienting_seq) + "." if len(center_orienting_seq) > 0 else "") + ".".join(mapped_sol)
print(mapped_sol)

print(f"Validating")
state = np.array(puzzle["initial_state"].split(";"))
for move_name in solution:
    state = state[moves[move_name]]

num_difference = evaluate_difference(state, solution_state)
wildcards = puzzle['num_wildcards']

current_solution = []
with open(f"{args.sol_dir}/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

if num_difference <= wildcards:
    print(f"Solution is valid. Diff to WC: {num_difference} <= {wildcards}")
    # Write it to the solution file
    if len(solution) < len(current_solution):
        print(f"New solution is shorter than current solution. Writing to file.")
        print(f"Length of new solution: {len(solution)}")
        print(f"Length of current solution: {len(current_solution)}")
        with open(f"data/solutions/{args.id}.txt", "w") as fp:
            fp.write(mapped_sol)
    else:
        print(f"New solution is longer than current solution.")
        print(f"Length of new solution: {len(solution)}")
        print(f"Length of current solution: {len(current_solution)}")
else:
    print(f"Solution is invalid. Diff to WC: {num_difference} > {wildcards}")
    print(f"Expected: {solution_state}")
    print(f"Got: {state}")
    print(f"Writing to partial solution file")

    faces = state_to_faces(state, n)
    print("GOT FACES")
    print_faces(faces, n)

    with open(f"data/reskin_partial_sol.txt", "w") as f:
        f.write(mapped_sol)
