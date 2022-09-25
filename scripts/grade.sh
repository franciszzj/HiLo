#!/bin/bash
# sh scripts/grade.sh

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/grade.py ./submission ./submission
