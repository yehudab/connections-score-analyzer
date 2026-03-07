#!/usr/bin/env python3
import os, psutil, sys
from scorer import compute_score_from_bar_image

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <image-path>")
    sys.exit(1)

score, status = compute_score_from_bar_image(sys.argv[1], save_debug=False)

# Instrumentation at the end
process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / 1024 / 1024
print(f"\n--- Resource Report ---")
print(f"Peak Memory Usage: {mem_mb:.2f} MB")
print(f"-----------------------")

if status == "success":
    print(score)
else:
    print(f"Error: {status}", file=sys.stderr)
    sys.exit(1)
