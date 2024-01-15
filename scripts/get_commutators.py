#! /usr/bin/env python3

import argparse
import subprocess
from util import *
import threading
import os

# ./scripts/get_commutators.py ./data/tws_phases/

def on_timeout(proc):
    print("Timed out")
    proc.kill()

def get_set_comms(puzdef, include, alg=3):
    twsearch_cmd = f"/Users/Win33/Documents/Programming/twsearch/build/bin/twsearch -A{alg} --microthreads 16 -M 32768"

    if include:
        # Read through the puzdef and find the sets
        sets = []
        with open(puzdef) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Set"):
                    sets.append(line.split()[1])

        # Create the omit string
        omit = " ".join([f"--omit {s}" for s in sets if s not in include.split()])

        twsearch_cmd += f" {omit}"

    twsearch_cmd += f" {puzdef}"

    print(twsearch_cmd)

    # Run twsearch
    proc = subprocess.Popen(twsearch_cmd.split(), stdout=subprocess.PIPE)

    print("Starting timer")
    timer = threading.Timer(1, on_timeout, (proc, ))
    timer.start()

    twsearch_folder = os.path.dirname(puzdef)
    comms_folder = os.path.join(twsearch_folder, "comms")

    # Read the output
    os.makedirs(comms_folder, exist_ok=True)

    file_name = (include if include else "all") + f"_{alg}" + ".txt"

    n_lines = 0
    lines = []
    try:
        while True:
            line = proc.stdout.readline().decode("utf-8").strip()
            # print(line)
            if not line:
                break
            n_lines += 1
            # print(line)
            lines.append(line)
    except KeyboardInterrupt:
        print("Interrupted")
        proc.kill()

    timer.cancel()

    # edges_line_count = 4783
    # expected_centers_line_count = 3103
    # expected_center_edges_line_count = 295
    # expected_middle_centers = 583

    # expected_lines = set()
    # expected_lines.add(edges_line_count)
    # expected_lines.add(expected_centers_line_count)
    # expected_lines.add(expected_center_edges_line_count)
    # expected_lines.add(expected_middle_centers)

    # print(f"Read {n_lines} lines")
    # assert n_lines in expected_lines, f"Expected {expected_lines} lines, got {n_lines}"

    with open(os.path.join(comms_folder, file_name), "w") as f:
        f.write("\n".join(lines))


parser = argparse.ArgumentParser(description='Get commutators from a puzdef')

parser.add_argument('puzdef', type=str, help='puzdef file')
parser.add_argument('--include', type=str, help='include these pieces')
parser.add_argument('--alg', type=int, default=3, help='twsearch algorithm')

args = parser.parse_args()

if args.include:
    print(f"Getting commutators for {args.include}")
    get_set_comms(args.puzdef, args.include, alg=args.alg)
else:
    print(f"Getting all commutators")

    # Read the puzdef and find the sets
    sets = []
    with open(args.puzdef) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Set"):
                line = line.split()

                # Skip "PIECE0" (the corners)
                # Skip the set with only 6 pieces (the centers)
                if line[1] != "PIECE0" and line[2] != "6":
                    sets.append(line[1])

    try:
        for set_name in sets:
            print(f"Getting commutators for {set_name}")
            get_set_comms(args.puzdef, set_name, alg=args.alg)
    except KeyboardInterrupt:
        print("Interrupted")