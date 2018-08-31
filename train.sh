#!/bin/bash

ls .gdrive && \
python3 train.py && \
epochTime=$(date +%s) && \
mv ./logs/001/trained_weights_stage_1.h5 ./training_${epochTime}.h5 && \
drive -c .gdrive upload --file training_${epochTime}.h5 && \
echo success && sleep 86400
