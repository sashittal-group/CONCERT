#!/bin/bash

BASE_DIR="../data/orig/simulation/bdt_ctmc"

for i in $(seq 0 24); do
    TEST_DIR="${BASE_DIR}/test_${i}"

    echo "Running inference for test_${i}..."

    ./bdt_ctmc.out \
        --num-states 3 \
        --num-trees 1 \
        --zero-death \
        --sampling-rate 0.5 \
        --transition-edges-file transition_edges.txt \
        "${TEST_DIR}" \
        --one-by-delta-t 1000 \
        --output "${TEST_DIR}/inference.out" \
        --log "${TEST_DIR}/inference.log"

    if [ $? -ne 0 ]; then
        echo "Failed for test_${i}"
    else
        echo "Completed test_${i}"
    fi
done