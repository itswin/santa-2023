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

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("--cube_move_names", action="store_true", default=False)

args = parser.parse_args()

puzzle = pd.read_csv("data/puzzles.csv").set_index("id").loc[args.id]
puzzle_type = puzzle["puzzle_type"].replace("/", "_")
print(f"puzzle_type: {puzzle_type}")

moves = get_moves(puzzle["puzzle_type"])
print(f"Number of moves: {len(moves)}")

solution_state = puzzle["solution_state"].split(";")

identity = np.arange(len(solution_state))

# Each piece starts in its own set
# Apply each move, if a piece is in the same set as another piece, merge the sets
# Repeat until no more sets are merged

sets, piece_to_set_index, set_to_sol_piece_to_index = decompose(solution_state, moves)

out = f"""Name {puzzle["puzzle_type"]}_decomposed

"""

set_str = """Set PIECE{set_num} {num_pieces} 1
"""

for set, pieces in sets.items():
    out += set_str.format(set_num=set, num_pieces=len(pieces))

solved_str = """
Solved
{SOLVED_SETS}
End
"""

solved_set_str = """
PIECE{set_num}
{solved}"""

solved_sets = ""
for set, pieces in sets.items():
    solved = " ".join([set_to_sol_piece_to_index[set][solution_state[piece]] for piece in pieces])
    solved_sets += solved_set_str.format(set_num=set, solved=solved)

out += solved_str.format(SOLVED_SETS=solved_sets)

move_str = """
Move {move_name}
{set_move_str}End
"""

set_move_str = """PIECE{set_num}
{set_move}
"""

n = int(puzzle_type.split("_")[-1])
inverse_move_map = get_inverse_move_map(n)

for name, move in moves.items():
    if args.cube_move_names:
        name = inverse_move_map[name]

    # Skip inverted moves
    if name.startswith("-"):
        continue

    set_moves = ""
    for set, pieces in sets.items():
        set_move = " ".join([str(piece_to_set_index[move[piece]]) for piece in pieces])

        # If the set move is the identity, don't include it
        if set_move == " ".join([str(i) for i in range(1, len(pieces) + 1)]):
            continue

        set_moves += set_move_str.format(set_num=set, set_move=set_move)
    out += move_str.format(move_name=name, set_move_str=set_moves)


twsearch_puzzles = f"./data/tws_phases/{puzzle_type}/"
Path(twsearch_puzzles).mkdir(parents=True, exist_ok=True)
with open(f"{twsearch_puzzles}/{puzzle_type}_{"decomposed" if not args.cube_move_names else "cube"}.tws", "w") as fp:
    fp.write(out)
