#!/usr/bin/env python3
# ./scripts/twsearch2x2.py --id 29 | ~/Documents/Programming/twsearch/build/bin/twsearch -M 32768 -q --shortenseqs ~/Documents/Programming/twsearch/samples/main/2x2x2_other.tws
import argparse
from util import *

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
parser.add_argument("--inv", action="store_true")
args = parser.parse_args()

with open(f"data/solutions/{args.id}.txt", "r") as fp:
    solution = fp.read().split(".")

if args.inv:
    solution = reversed(list(map(invert, solution)))

with open(f"/Users/Win33/Documents/Programming/twsearch/moves.txt", "w") as fp:
    fp.write(" ".join(solution))

print(" ".join(solution))
