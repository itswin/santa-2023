#!/usr/bin/env python3

from collections import deque
import itertools

left_str  = '''    a[{index}] = SCS_getAlg{i}c({static}, {left_rest}) + " " + SCS_getAlg{j}c({static}, {right_rest});'''
right_str = '''    a[{index}] = SCS_getAlg{j}c({left_rest}, {static}) + " " + SCS_getAlg{i}c({right_rest}, {static});'''

n = 23
index = 1
for i in range((n + 1) // 2 - ((n % 3) // 2), (n + 1) // 2 - 4, -2):
    j = n - i + 1
    rot = list(f"i{x}" for x in range(1, n + 1))
    d = deque(rot)

    # Skip 3. 5 cycles should be found.
    if i == 3 and n >= 13:
        continue

    # 15 and 17 databases are small.
    # Don't use them to reduce Wall time.
    if (i == 17 or j == 17):
        continue

    if (i == 15 or j == 15):
        continue

    for k in range(n):
        d = deque(rot)
        d.rotate(k)
        # print(d, k)
        static = d.pop()
        left_rest = list(itertools.islice(d, 0, i - 1))
        right_rest = list(itertools.islice(d, i - 1, n))
        left_rest = ", ".join(left_rest)
        right_rest = ", ".join(right_rest)

        print(left_str.format(index=index, i=i, j=j, static=static, left_rest=left_rest, right_rest=right_rest))
        index += 1
    
    for k in range(n):
        d = deque(rot)
        d.rotate(k)
        # print(d, k)
        static = d.pop()
        left_rest = list(itertools.islice(d, 0, j - 1))
        right_rest = list(itertools.islice(d, j - 1, n))
        left_rest = ", ".join(left_rest)
        right_rest = ", ".join(right_rest)

        print(right_str.format(index=index, i=i, j=j, static=static, left_rest=left_rest, right_rest=right_rest))
        index += 1

