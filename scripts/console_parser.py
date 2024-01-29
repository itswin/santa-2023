#!/usr/bin/env python3
import argparse
import pandas as pd
import itertools
import numpy as np
import subprocess
from util import *

# seq 160 160 | xargs -P 4 -I {} sh -c "python3 scripts/555_solve.py {}"

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)

args = parser.parse_args()

orbit_algs = []

with open(args.file, "r") as fp:
    for line in fp.readlines():
        if "alg:" in line:
            alg = line.split("alg:")[1].strip()
            # print(alg)
            # print(len(alg.split()))
            # print()
            orbit_algs.append(alg)
        elif "Alg:" in line:
            alg = line.split("Alg:")[1].strip()
            # print(alg)
            # print(len(alg.split()))
            # print()
            orbit_algs.append(alg)

full_alg = " ".join(orbit_algs)
with open("data/SSE/console_alg.txt", "w") as fp:
    fp.write(full_alg)

print(f"Length: {len(full_alg.split())}")
print(f"Num orbits: {len(orbit_algs)}")
