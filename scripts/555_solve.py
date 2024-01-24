#!/usr/bin/env python3
import argparse
import pandas as pd
import itertools
import numpy as np
import subprocess
from util import *

# seq 160 160 | xargs -P 4 -I {} sh -c "python3 scripts/555_solve.py {}"

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("--sol_dir", type=str, default="data/solutions")
parser.add_argument("--out_sol_dir", type=str, default="data/solutions")
parser.add_argument("--partial_sol", type=str, default=None)
parser.add_argument("--show_faces_only", action="store_true", default=False)
parser.add_argument("--reskin", action="store_true", default=False)

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
print(state)

print("INITIAL", state)

move_map = get_move_map(n)
print(move_map)

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

if args.reskin:
    edges = get_edges(n, skip=2)
    odd_centers = get_diff_odd_centers(n)

    print(f"Edges {len(edges)}", edges)
    print(odd_centers)

    n2 = n ** 2
    NORMAL_SOLUTION = "A" * n2 + "B" * n2 + "C" * n2 + "D" * n2 + "E" * n2 + "F" * n2

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

print_faces(faces, n)

cubestring = make_cubestring(faces)
print(cubestring)

if args.show_faces_only:
    exit()

directory = "/Users/Win33/Documents/Programming/cube555/"
SOLVER_PATH = " make cmd".split()
cmd = SOLVER_PATH + [f"ARGS={cubestring}"]
# cmd = ["cat", "scripts/split.py"]

out = subprocess.check_output(cmd, cwd=directory)

out = out.decode("utf-8").strip()
out = out.split("\n")

# Search for the solution line
for line in out:
    if line.startswith("Solution: "):
        sol = line.split(":")[1].strip()
        break

print(sol)

# Map it back to our move set
mapped_sol = []
for move in sol.split():
    mapped_sol.append(move_map[move])

mapped_sol = center_orienting_seq + mapped_sol
formatted_sol = ".".join(mapped_sol)
mapped_sol = formatted_sol.split(".")
print(formatted_sol)

current_solution = []
with open(f"{args.sol_dir}/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

wildcards = puzzle['num_wildcards']

print(f"Validating")
state = np.array(puzzle["initial_state"].split(";"))
for move_name in mapped_sol:
    state = state[moves[move_name]]

num_difference = evaluate_difference(state, solution_state)

if num_difference <= wildcards:
    print(f"Solution is valid. Diff to WC: {num_difference} <= {wildcards}")

    # Write it to the solution file
    with open(f"{args.out_sol_dir}/{args.id}.txt", "w") as fp:
        fp.write(formatted_sol)

    print(f"Running group theory and k-moves. Length before: {len(mapped_sol)}")

    # Run group_theory.py and k_moves.py
    subprocess.run(["python3", "scripts/group_theory.py", str(args.id)])
    subprocess.run(["python3", "scripts/k_moves.py", str(args.id)])

    # Reread the solution
    with open(f"{args.out_sol_dir}/{args.id}.txt", "r") as fp:
        mapped_sol = fp.read().split(".")

    # Write it to the solution file
    if len(mapped_sol) < len(current_solution):
        print(f"New solution is shorter than current solution. Writing to file.")
        print(f"Length of new solution: {len(mapped_sol)}")
        print(f"Length of current solution: {len(current_solution)}")
        with open(f"{args.out_sol_dir}/{args.id}.txt", "w") as fp:
            fp.write(formatted_sol)
    else:
        print(f"New solution is longer than current solution.")
        print(f"Length of new solution: {len(mapped_sol)}")
        print(f"Length of current solution: {len(current_solution)}")

        print(f"Writing to partial solution file")
        with open(f"data/tpr_partial_sol.txt", "w") as f:
            f.write(formatted_sol)

        # Rewrite the old current solution
        with open(f"{args.out_sol_dir}/{args.id}.txt", "w") as fp:
            fp.write(".".join(current_solution))
else:
    print(f"Solution is invalid. Diff to WC: {num_difference} > {wildcards}")
    print(f"Expected: {solution_state}")
    print(f"Got: {state}")
    print(f"Writing to partial solution file")

    with open(f"data/tpr_partial_sol.txt", "w") as f:
        f.write(formatted_sol)

