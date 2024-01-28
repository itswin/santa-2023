#!/usr/bin/env python3

import random

l = []
for i in range(24):
    l.append(chr(97+i).upper())
print(l)

r = random.sample(l,23)
print(f"({"".join(r)})")
