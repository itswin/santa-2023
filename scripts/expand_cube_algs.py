#!/usr/bin/env python3
import argparse
import pandas as pd
from util import *
import re

def expand_move_names(moves):
    # If F and B are in the move, replace them with (F,B) (U,D) (L,R) (B,F) (D,U) (R,L)
    new_moves = []

    valid_map = {
        "f": ["F", "B", "U", "D", "L", "R"],
        "d": ["B", "F", "U", "D", "L", "R"],
        "r": ["U", "D", "L", "R", "F", "B"],
    }
    
    opp_map = {
        "F": "B",
        "B": "F",
        "U": "D",
        "D": "U",
        "L": "R",
        "R": "L",
    }

    moves = moves.lower()
    for (f, d, r) in itertools.product(valid_map["f"], valid_map["d"], valid_map["r"]):
        valid_map = {
            "f": f,
            "d": d,
            "r": r,
            "u": opp_map[d],
            "b": opp_map[f],
            "l": opp_map[r],
        }

        new_moves.append(moves.replace("f", valid_map["f"])
                            .replace("d", valid_map["d"])
                            .replace("r", valid_map["r"])
                            .replace("u", valid_map["u"])
                            .replace("b", valid_map["b"])
                            .replace("l", valid_map["l"]))
        

    # print(moves)
    # print(new_moves)

    return new_moves

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
puzzle_type = puzzle["puzzle_type"].replace("/", "_")

moves = get_moves(puzzle["puzzle_type"])

n = int(puzzle_type.split("_")[-1])
move_map = get_move_map(n)

algs_7_file = "./data/tws_phases/cube_7_7_7/algs.txt"

algs_7 = []

with open(algs_7_file, 'r') as f:
    algs = f.readlines()
    for line in algs:
        if line == "\n":
            continue

        line = line.strip()
        new_algs = expand_move_names(line)
        algs_7 += new_algs

normal_moves = ["U", "D", "L", "R", "F", "B"]

mid_layer_moves = []
for i in range(2,4):
    mid_layer_moves += [str(i) + move for move in normal_moves]

print(mid_layer_moves)

algs = set()

for alg in algs_7:
    # If a middle layer move exists in the alg,
    # range through all possible middle layer moves of that type
    # and add the resulting algs to the set

    # Find all distinct mid_layer_moves that exist in the alg
    alg_moves = set()
    for mid_layer_move in mid_layer_moves:
        for move in alg.split():
            if mid_layer_move in move:
                alg_moves.add(mid_layer_move)

    # print(alg, alg_moves)
    # If there are no mid_layer_moves in the alg, add it to the set
    if len(alg_moves) == 0:
        algs.add(alg)
        continue

    # Otherwise, range through all possible middle layer moves of that type
    # and add the resulting algs to the set
    if len(alg_moves) == 1:
        alg_move = alg_moves.pop()
        move_format = "{n}" + alg_move[-1]
        for i in range(2, n // 2 + 1):
            algs.add(alg.replace(alg_move, move_format.format(n=i)))
    elif len(alg_moves) == 2:
        alg_move1, alg_move2 = alg_moves
        move_format1 = "{n}" + alg_move1[-1].lower()
        move_format2 = "{n}" + alg_move2[-1].lower()

        for i in range(2, n // 2 + 1):
            for j in range(2, n // 2 + 1):
                algs.add(alg.replace(alg_move1, move_format1.format(n=i)).replace(alg_move2, move_format2.format(n=j)).upper())
    else:
        # Otherwise, all middle layers should be the same?
        n_val = next(iter(alg_moves))[0]
        for move in alg_moves:
            assert move[0] == n_val

        for i in range(2, n // 2 + 1):
            for move in alg_moves:
                new_alg = new_alg.replace(move, "{n}" + move[-1]).format(n=i)

        algs.add(new_alg.upper())

print(f"Found {len(algs)} algs")
# print(algs)

# Write the algs to a file
twsearch_puzzle = f"./data/tws_phases/cube_{n}_{n}_{n}"
Path(twsearch_puzzle).mkdir(parents=True, exist_ok=True)

with open(twsearch_puzzle + "/exp_algs.txt", 'w') as f:
    f.write("\n".join(algs))
