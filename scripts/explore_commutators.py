#! /usr/bin/env python3

import argparse
from collections import Counter
import pandas as pd
import itertools
import numpy as np
import subprocess
from util import *
from typing import List

def get_affecting_commutators(idx: int, commutators: List[Commutator]) -> List[Commutator]:
    return [comm for comm in commutators if comm.move[idx] != idx]

def find_overlapping_commutators(idx1, idx2, commutators):
    comm_idx1 = get_affecting_commutators(idx1, commutators)
    comm_idx2 = get_affecting_commutators(idx2, commutators)

    print(f"Number of commutators affecting {idx1}: {len(comm_idx1)}")
    print(f"Number of commutators affecting {idx2}: {len(comm_idx2)}")

    overlap = []
    for comm1 in comm_idx1:
        affected_indices = set([i for i in range(len(comm1.move)) if comm1.move[i] != i and i != idx1])
        for comm2 in comm_idx2:
            affected_indices2 = set([i for i in range(len(comm2.move)) if comm2.move[i] != i and i != idx2])
            if affected_indices == affected_indices2:
                overlap.append((comm1.name, comm2.name, len(affected_indices), affected_indices))
                print(comm1.name, comm2.name, len(affected_indices), affected_indices)

    print(f"Number of overlapping commutators: {len(overlap)}. For {idx1} and {idx2}")

    return overlap

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("conjugate_file", type=str)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
puzzle_type = puzzle["puzzle_type"].replace("/", "_")

solution_state = np.array(puzzle["solution_state"].split(";"))

moves = get_moves(puzzle["puzzle_type"])

# commutators = read_conjugates(args.conjugate_file, moves, None, 25, False)
# print(f"Number of conjugates: {len(commutators)}")

idx1 = 133
idx2 = 172

# overlap = find_overlapping_commutators(idx1, idx2, commutators)
# overlap2 = find_overlapping_commutators(8, 9, commutators)

# for o1 in overlap:
#     for o2 in overlap2:
#         if o1[3] == o2[3]:
#             print("Overlapping", o1, o2)

# (f1|f49,[r1 f0 -r1,f33]) (f41|f45,[r2 f7 -r2,f40]) 2 [130, 131]
# (f1|f49,[r1 f0 -r1,f33]) (f41|f45,[r2 f7 -r2,f40]) 2 [130, 131]

comm = Conjugate("(f1|f49,[r1 f0 -r1,f33])", moves)
affected = [i for i in range(len(comm.move)) if comm.move[i] != i]
print(comm.move)
print(affected)

comm2 = Conjugate("(f41|f45,[r2 f7 -r2,f40])", moves)
affected2 = [i for i in range(len(comm2.move)) if comm2.move[i] != i]
print(comm2.move)
print(affected2)

composed = comm.invert().compose(comm2.invert())
affected3 = [i for i in range(len(composed.move)) if composed.move[i] != i]
print(composed.move)
print(affected3)

solution_state = solution_state[moves["r0"]]
print(solution_state)
