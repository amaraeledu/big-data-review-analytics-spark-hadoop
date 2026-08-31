#!/usr/bin/env python3
"""
reducer.py - Hadoop Streaming reducer for the MapReduce comparison job.

Sums the (product, 1) pairs emitted by mapper.py to produce a total
mention count per product, mirroring a classic word-count pattern
applied to product mentions.
"""

import sys

current_product = None
current_count = 0

for line in sys.stdin:
    product, count = line.rstrip("\n").split("\t", 1)
    count = int(count)

    if current_product == product:
        current_count += count
    else:
        if current_product is not None:
            print(f"{current_product}\t{current_count}")
        current_product = product
        current_count = count

# emit the last product
if current_product is not None:
    print(f"{current_product}\t{current_count}")
