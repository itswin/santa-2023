#!/usr/bin/env python3
import argparse
import pandas as pd
import itertools
import numpy as np
from subprocess import Popen, PIPE
import subprocess
from util import *
from os import listdir
from os.path import isfile, join

# seq 210 234 | xargs -I {} python3 scripts/twsearch_phases.py {}
# cat moves.txt | time /Users/Win33/Documents/Programming/twsearch/build/bin/twsearch -q -s --microthreads 16 -M 32768 ./data/tws_phases/globe_6_4/globe_6_4_phase6.tws

def evaluate_difference(current_state, final_state):
    return np.count_nonzero(current_state != final_state)

def get_enumerated_phase_list(base_string, num_phases):
    return [base_string + str(i) + ".tws" for i in range(1, num_phases + 1)]

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("--sol_dir", type=str, default="data/solutions")
parser.add_argument("--out_sol_dir", type=str, default="data/solutions")
parser.add_argument("--moves", action="store_true", default=False)
parser.add_argument("--unique", action="store_true", default=False)
parser.add_argument("--commutator_file", type=str, default=None)
parser.add_argument("--pbp", type=int, default=None)
parser.add_argument("--subdir", type=str, default=None)
parser.add_argument("--twsearch_path", type=str, default="../twsearch/build/bin/twsearch")
parser.add_argument("--twsearch_mem", type=int, default=24576)
parser.add_argument("--kaggle", action="store_true", default=False)
parser.add_argument("--phases_file", type=str, default="default.txt")

args = parser.parse_args()

if args.kaggle:
    args.twsearch_path = "/kaggle/twsearch/build/bin/twsearch"
    args.out_sol_dir = "/kaggle/working"
    cache_dir = "/kaggle/working/twsearch_cache"
    SOLVER_CMD = f"{args.twsearch_path} --writeprunetables always --microthreads 16 -q -s -M {args.twsearch_mem} --cachedir {cache_dir}"
else:
    SOLVER_CMD = f"{args.twsearch_path} --writeprunetables always --microthreads 16 -q -s -M {args.twsearch_mem}"

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
print(puzzle)

puzzle_type = puzzle["puzzle_type"]
n = int(puzzle_type.split("/")[-1])
moves = get_moves(puzzle["puzzle_type"])
print(f"Number of moves: {len(moves)}")

initial_state = puzzle["initial_state"].split(";")
solution_state = puzzle["solution_state"].split(";")

unique = solution_state[0].startswith("N")
if unique:
    puzzle_type += "_unique"

puzzle_type = puzzle_type.replace("/", "_")

if args.pbp:
    write_piece_phases(puzzle, args.pbp)
    puzzle_type += "_pbp"
else:
    write_tws_file(puzzle, unique)

# Use the current solution as a scramble
with open(f"data/solutions/{args.id}.txt", "r") as fp:
    current_solution = fp.read().split(".")

is_move_cyclic = {}
identity = np.arange(len(initial_state))
for name, move in moves.items():
    m = move[move]
    is_move_cyclic[name] = (m == identity).all()

def invert_if_not_cycle(move):
    if move[0] == '-':
        return move[1:]
    elif is_move_cyclic[move]:
        return move
    else:
        return "-" + move

scramble = " ".join(reversed(list(map(invert_if_not_cycle, current_solution))))
print(scramble)

if args.moves:
    # with open("/Users/Win33/Documents/Programming/twsearch/moves.txt", "w") as fp:
    #     fp.write(scramble)
    with open("./moves.txt", "w") as fp:
        fp.write(scramble)
    exit()

twsearch_puzzles = f"./data/tws_phases/{puzzle_type}/"

phases = None
if args.pbp:
    puzzle_type += str(args.pbp)
    phases = get_enumerated_phase_list(puzzle_type + "_", len(initial_state) - 1)
else:
    try:
        phases = get_phase_list(twsearch_puzzles + args.phases_file)
    except:
        print(f"Couldn't find phase list file: {twsearch_puzzles + args.phases_file}. Please supple --phases_file argument.")
        exit()

solution_so_far = []

for tws_file in phases:
    print(f"Running {tws_file}")
    SOLVER_PATH = f"{SOLVER_CMD} {twsearch_puzzles}{tws_file}".split()
    p = Popen(SOLVER_PATH, stdout=PIPE, stdin=PIPE, stderr=PIPE)
    p.stdin.write(scramble.encode("utf-8"))
    p.stdin.write(b"\n")
    p.stdin.flush()

    # Search for the solution line
    sol = None
    last_depth = None
    writing = False
    while True:
        try:
            if p.poll() is not None:
                break
            line = p.stdout.readline().decode("utf-8").strip()
            if "FOUND SOLUTION: " in line:
                sol = line.split(":")[1].strip()
                if sol is None:
                    print("No solution needed. Skipping phase")
                    sol = ""
                print(f"\tPartial Solution: {sol}")
                break
            if line.startswith("Depth"):
                last_depth = line.split()[1].strip()
                clear_line()
                print(f"\tLast depth: {last_depth}", end='\r')
            elif "Writing" in line:
                if "written" in line:
                    clear_line()
                    print(f"\tLast depth: {last_depth}. Write complete1.", end='\r')
                else:
                    clear_line()
                    print(f"\tLast depth: {last_depth}. WRITING. DO NOT INTERRUPT", end='\r')
                    writing = True
            elif "Written in" in line:
                clear_line()
                print(f"\tLast depth: {last_depth}. Write complete2.", end='\r')
                writing = False
        except:
            if writing:
                print("Writing. Do not interrupt")
            else:
                print("Interrupted")
                p.kill()
                break

    if sol == "":
        continue
    elif sol is None:
        print("No solution found. Exiting")
        print("Last log line: ", line)
        print("Error: ")
        while True:
            err_line = p.stderr.readline().decode("utf-8").strip()
            if err_line == "":
                break
            print(err_line)

        exit()

    partial_sol = sol.split(".")
    solution_so_far += partial_sol
    print("\tSolution so far: ", ".".join(solution_so_far))
    scramble += " " + " ".join(partial_sol)

print(f"Validating")
state = np.array(puzzle["initial_state"].split(";"))
for move_name in solution_so_far:
    state = state[moves[move_name]]

differences = evaluate_difference(state, solution_state)
wildcards = puzzle['num_wildcards']
if differences <= wildcards:
    print(f"Solution is valid. Diff to WC: {differences} <= {wildcards}")
else:
    print("Solution is invalid")
    print(f"Expected: {solution_state}")
    print(f"Got: {state}")
    assert False

if len(solution_so_far) < len(current_solution):
    print(f"New solution is shorter than current solution. Writing to file.")
    with open(f"{args.out_sol_dir}/{args.id}.txt", "w") as fp:
        fp.write(".".join(solution_so_far))
else:
    print(f"New solution is longer than current solution.")
    print(f"Length of new solution: {len(solution_so_far)}")
    print(f"Length of current solution: {len(current_solution)}")
