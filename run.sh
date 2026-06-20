#!/bin/bash
# Usage:
#   ./run.sh eval pkg_000005          — classify + accuracy check
#   ./run.sh pipeline pkg_000005      — full pipeline (classify + segment + stitch)
#   ./run.sh eval                     — run all 40 packages

export OPENAI_API_KEY="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"

MODE="${1:-eval}"
PKG="$2"

if [ "$MODE" = "eval" ]; then
    if [ -n "$PKG" ]; then
        python3 src/classification/eval_classify.py --pkg "DataSet /$PKG"
    else
        python3 src/classification/eval_classify.py
    fi
elif [ "$MODE" = "pipeline" ]; then
    if [ -z "$PKG" ]; then
        echo "Usage: ./run.sh pipeline pkg_000005"
        exit 1
    fi
    python3 src/pipeline/run_pipeline.py --pkg "DataSet /$PKG"
else
    echo "Usage:"
    echo "  ./run.sh eval pkg_000005      — classify + accuracy"
    echo "  ./run.sh pipeline pkg_000005  — full pipeline"
    echo "  ./run.sh eval                 — all 40 packages"
fi
