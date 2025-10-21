#!/usr/bin/env python3
import sys
from pathlib import Path
from collections import defaultdict

def remove_empty_dirs(base_dir):
    """Remove empty subdirectories under base_dir."""
    removed = []
    for dirpath in sorted(base_dir.rglob('*'), reverse=True):
        if dirpath.is_dir() and dirpath != base_dir:
            try:
                dirpath.rmdir()
                removed.append(dirpath)
            except OSError:
                pass
    return removed

def get_key(rel_path):
    """Generate lookup key from relative path, stripping extensions."""
    path_str = str(rel_path)
    if path_str.lower().endswith('.jsonl.gz'):
        return path_str[:-9].lower()
    return str(rel_path.with_suffix('')).lower()

def print_counts(pdf_count, md_count, jsonl_count):
    """Print file counts."""
    print(f"  MD       : {md_count}")
    print(f"  PDF      : {pdf_count}")
    print(f"  JSONL.GZ : {jsonl_count}")

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} /path/to/documents")
    sys.exit(1)

base_dir = Path(sys.argv[1]).resolve()

if not base_dir.is_dir():
    print(f"Error: '{base_dir}' is not a directory.")
    sys.exit(2)

print(f"Scanning under: {base_dir}\n")

# Collect all files and build lookup
files_by_path = defaultdict(lambda: {"pdf": None, "md": None, "jsonl": None})
pdf_files = []
md_files = []
jsonl_files = []

for path in base_dir.rglob("*"):
    if not path.is_file():
        continue
    
    rel_path = path.relative_to(base_dir)
    suffix = path.suffix.lower()
    
    if suffix == ".pdf":
        key = get_key(rel_path)
        files_by_path[key]["pdf"] = path
        pdf_files.append(path)
    elif suffix == ".md":
        key = get_key(rel_path)
        files_by_path[key]["md"] = path
        md_files.append(path)
    elif path.name.lower().endswith(".jsonl.gz"):
        key = get_key(rel_path)
        files_by_path[key]["jsonl"] = path
        jsonl_files.append(path)

# Find orphans
to_delete = []
for key, files in files_by_path.items():
    if files["pdf"] and not files["md"]:
        to_delete.append(files["pdf"])
        if files["jsonl"]:
            to_delete.append(files["jsonl"])
    elif files["jsonl"] and not files["pdf"] and not files["md"]:
        to_delete.append(files["jsonl"])

if not to_delete:
    print("No candidates found.")
else:
    print(f"Files to delete ({len(to_delete)}):")
    for f in sorted(to_delete):
        print(f)
    print()
    answer = input(f"Delete these {len(to_delete)} files? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted.")
        sys.exit(0)
    
    # Track what gets deleted
    deleted_pdfs = sum(1 for f in to_delete if f in pdf_files)
    deleted_jsonl = sum(1 for f in to_delete if f in jsonl_files)
    
    for f in to_delete:
        f.unlink()
    
    print(f"Deleted {len(to_delete)} files.")
    
    # Update counts
    pdf_files = [f for f in pdf_files if f not in to_delete]
    jsonl_files = [f for f in jsonl_files if f not in to_delete]
    
    # Remove empty directories
    print("\nRemoving empty subdirectories...")
    removed = remove_empty_dirs(base_dir)
    for d in removed:
        print(f"Removed: {d}")

print("\nFinal counts:")
print_counts(len(pdf_files), len(md_files), len(jsonl_files))

# Check if PDFs remain and offer to delete all
if pdf_files:
    print(f"\n{len(pdf_files)} PDF files remain.")
    answer = input("Delete all PDFs? [y/N] ").strip().lower()
    if answer in ("y", "yes"):
        for pdf in pdf_files:
            pdf.unlink()
        print(f"Deleted {len(pdf_files)} PDF files.")
        pdf_files.clear()
        
        # Remove empty directories
        print("\nRemoving empty subdirectories...")
        removed = remove_empty_dirs(base_dir)
        for d in removed:
            print(f"Removed: {d}")
        
        print("\nFinal counts after PDF deletion:")
        print_counts(len(pdf_files), len(md_files), len(jsonl_files))