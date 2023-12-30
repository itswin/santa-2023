#!/usr/bin/env python3
import argparse
import pandas as pd
import os
import numpy as np
from util import *

parser = argparse.ArgumentParser()
parser.add_argument("list", type=str)
parser.add_argument("to", type=int)

args = parser.parse_args()

l = args.list.split(" ")
m = map(lambda _: str(args.to), l)
print(" ".join(m))
