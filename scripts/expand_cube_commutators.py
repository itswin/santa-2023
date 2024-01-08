#!/usr/bin/env python3
import argparse
import pandas as pd
from util import *
import re

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
puzzle_type = puzzle["puzzle_type"].replace("/", "_")

moves = get_moves(puzzle["puzzle_type"])

n = int(puzzle_type.split("_")[-1])
move_map = get_move_map(n)

comms_7_file = "./data/tws_phases/cube_7_7_7/comms.txt"

comms_7 = []

with open(comms_7_file, 'r') as f:
    comms = f.readlines()

    commutator_re = re.compile("(\\[.*\\])")
    commutator_re = re.compile("\\[(.*),(.*)\\]")
    for line in comms:
        match = commutator_re.search(line)
        if match:
            comms_7.append((match.group(1).split(), match.group(2).split()))

normal_moves = ["U", "D", "L", "R", "F", "B"]

mid_layer_moves = []
for i in range(2,4):
    mid_layer_moves += [str(i) + move for move in normal_moves]

print(mid_layer_moves)

commutators = set()

# Comm is a [X, Y] pair
for comm in comms_7:
    # If a middle layer move exists in the commutator,
    # range through all possible middle layer moves of that type
    # and add the resulting commutators to the set

    # Find all distinct mid_layer_moves that exist in the commutator
    comm_moves = set()
    for mid_layer_move in mid_layer_moves:
        for move in itertools.chain(comm[0], comm[1]):
            if mid_layer_move in move:
                comm_moves.add(mid_layer_move)

    # If there are no mid_layer_moves in the commutator, add it to the set
    if len(comm_moves) == 0:
        # Reformat the commutator
        comm = "[" + " ".join(comm[0]) + "," + " ".join(comm[1]) + "]"
        commutators.add(comm)
        continue

    # Otherwise, range through all possible middle layer moves of that type
    # and add the resulting commutators to the set
    if len(comm_moves) == 1:
        comm_move = comm_moves.pop()
        move_format = "{n}" + comm_move[-1]
        comm = "[" + " ".join(comm[0]) + "," + " ".join(comm[1]) + "]"
        for i in range(2, n // 2 + 1):
            commutators.add(comm.replace(comm_move, move_format.format(n=i)))
    elif len(comm_moves) == 2:
        comm_move1, comm_move2 = comm_moves
        move_format1 = "{n}" + comm_move1[-1].lower()
        move_format2 = "{n}" + comm_move2[-1].lower()

        comm = "[" + " ".join(comm[0]) + "," + " ".join(comm[1]) + "]"
        for i in range(2, n // 2 + 1):
            for j in range(2, n // 2 + 1):
                commutators.add(comm.replace(comm_move1, move_format1.format(n=i)).replace(comm_move2, move_format2.format(n=j)).upper())
    else:
        # Otherwise, all middle layers should be the same?
        n_val = next(iter(comm_moves))[0]
        for move in comm_moves:
            assert move[0] == n_val

        for i in range(2, n // 2 + 1):
            new_comm = "[" + " ".join(comm[0]) + "," + " ".join(comm[1]) + "]"
            for move in comm_moves:
                new_comm = new_comm.replace(move, "{n}" + move[-1]).format(n=i)

        commutators.add(new_comm.upper())

print(f"Found {len(commutators)} commutators")

# Write the commutators to a file
twsearch_puzzle = f"./data/tws_phases/cube_{n}_{n}_{n}"
Path(twsearch_puzzle).mkdir(parents=True, exist_ok=True)

with open(twsearch_puzzle + "/comms.txt", 'w') as f:
    f.write("\n".join(commutators))
