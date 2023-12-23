#!/usr/bin/env python3
# ./scripts/twsearch2x2.py --id 29 | ~/Documents/Programming/twsearch/build/bin/twsearch -M 32768 -q --shortenseqs ~/Documents/Programming/twsearch/samples/main/2x2x2_other.tws
import argparse
from get_moves import get_moves

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, required=True)
args = parser.parse_args()

with open(f"data/solutions/{args.id}.txt", "r") as fp:
    solution = fp.read().split(".")

with open(f"data/tmp_{args.id}.txt", "w") as fp:
    fp.write(" ".join(solution))

print(" ".join(solution))
