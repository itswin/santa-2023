#!/usr/bin/env python3
import argparse
import pandas as pd
import itertools
import numpy as np
from subprocess import Popen, PIPE
import subprocess
from util import *

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
puzzle_type = puzzle["puzzle_type"].replace("/", "_")

m = int(puzzle_type.split("_")[1])
n = int(puzzle_type.split("_")[2])

moves = get_moves(puzzle["puzzle_type"])

commutators = []

# m = 6, n = 16
# [r0 -r6,f0] ... [r0 -r6,f15]
# [-r0 r6,f0] ... [-r0 r6,f15]
move_formats = ["[r{m} -r{n},f{i}]", "[-r{m} r{n},f{i}]"]
for move_format in move_formats:
    for i in range(m // 2):
        m_x = i
        n_x = m - i
        for j in range(2 * n):
            move = move_format.format(m=m_x, n=n_x, i=j)
            commutators.append(move)

move_formats = []
# [r0 f0 -r0,f8] ... [r0 f15 -r0,f7]
move_formats.append(["[r{m} f{i} -r{m},f{j}]"])
# [-r0 f0 r0,f8] ... [-r0 f15 r0,f7]
move_formats.append(["[-r{m} f{i} r{m},f{j}]"])

# [r0 f0 r6,f0] ... [r0 f15 r6,f15]
move_formats.append(["[r{m} f{i} r{n},f{i}]"])
# [r6 f0 r0,f0] ... [r6 f15 r0,f15] (Covered by ranging m 0-6)

# [-r0 f0 -r6,f0] ... [-r0 f15 -r6,f15]
move_formats.append(["[-r{m} f{i} -r{n},f{i}]"])
# [-r6 f0 -r0,f0] ... [-r6 f15 -r0,f15] (Covered by ranging m 0-6)

for move_format in move_formats:
    for i in range(m):
        m_x = i
        n_x = m - i
        for i in range(2 * n):
            j = (i + n) % (2 * n)
            move = move_format[0].format(m=m_x, n=n_x, i=i, j=j)
            commutators.append(move)

print(f"Number of commutators: {len(commutators)}")

# Write them to a file
twsearch_folder = f"data/tws_phases/{puzzle_type}"

commutator_file = f"{twsearch_folder}/commutators.txt"
with open(commutator_file, "w") as fp:
    for comm in commutators:
        fp.write(comm + "\n")
