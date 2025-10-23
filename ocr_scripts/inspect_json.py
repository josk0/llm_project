import gzip
import json
import random
import sys

if len(sys.argv) < 2:
    print("Usage: python script.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]

# Read first object
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    first_line = f.readline()
    first_obj = json.loads(first_line)
    
print("Top-level keys:")
print(list(first_obj.keys()))

print("\nFirst object structure:")
for key, value in first_obj.items():
    print(f"  {key}: {type(value).__name__}", end="")
    if isinstance(value, (list, dict)):
        print(f" (length: {len(value)})")
    else:
        print()

print("\nFirst object preview:")
print(json.dumps(first_obj, indent=2)[:2000])

# Count total lines
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    line_count = sum(1 for _ in f)
print(f"\nTotal objects: {line_count:,}")

# Sample 3 random objects to check consistency
print("\nSampling 3 random objects...")
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    lines = [line for _, line in zip(range(10000), f)]  # load first 10k to sample from
    
for i, line in enumerate(random.sample(lines, min(3, len(lines)))):
    obj = json.loads(line)
    print(f"Sample {i+1} keys: {list(obj.keys())}")