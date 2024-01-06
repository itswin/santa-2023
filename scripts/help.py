#! /usr/bin/env python3
import re
import argparse

parser = argparse.ArgumentParser(description='Converts a file from one format to another.')
parser.add_argument('input', type=str, help='Input file')

args = parser.parse_args()

with open(args.input, 'r') as f:
    lines = f.readlines()
    num_conj = 0
    num_comms = 0
    other = 0
    for line in lines:
        if line[0] == '(':
            num_conj += 1
        elif line[0] == '[':
            num_comms += 1
        else:
            other += 1
            print(line)

    print(f"Conjugates: {num_conj}")
    print(f"Commutators: {num_comms}")
    print(f"Other: {other}")
    print(f"Total: {num_conj + num_comms + other}")
