#!/usr/bin/env bash

if [ ! $# -eq 1 ]; then
    echo "Usage: $0 <submit_message>"
    exit 1
fi

file="data/submission.csv"

if test -f "$file"; then
    kaggle competitions submit -c santa-2023 -f $file -m "$1"
    rm submission.tar.gz
else
    echo "Could not find $file."
    exit 1
fi
