#!/usr/bin/env python3
"""
mapper.py - Hadoop Streaming mapper for the MapReduce comparison job.

Reads each line of amazon_sample.tsv from stdin, extracts the product
field, and emits a (product, 1) key-value pair for downstream counting
by reducer.py. Run via Hadoop Streaming:

    hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
        -input /Data/amazon_sample.tsv \
        -output /Output/mapreduce_results \
        -mapper /Data/mapper.py \
        -reducer /Data/reducer.py
"""

import sys

for line in sys.stdin:
    fields = line.rstrip("\n").split("\t")
    if not fields or fields[0] == "product_id":
        # skip header row
        continue

    product = fields[0].strip()
    if product:
        print(f"{product}\t1")
