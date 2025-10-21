#!/usr/bin/env python3
import os
import gzip
import json
from pathlib import Path

def process_folder(documents_dir):
    documents_path = Path(documents_dir)
    
    if not documents_path.exists():
        print(f"Error: Directory {documents_dir} does not exist")
        return
    
    # Collect all .md and .jsonl.gz files by subfolder
    for root, dirs, files in os.walk(documents_path):
        root_path = Path(root)
        
        # Group files by stem
        md_files = {}
        jsonl_files = {}
        
        for file in files:
            file_path = root_path / file
            
            if file.endswith('.md'):
                stem = file[:-3]
                md_files[stem] = file_path
            elif file.endswith('.jsonl.gz'):
                stem = file[:-9]
                jsonl_files[stem] = file_path
        
        # Process pairs
        all_stems = set(md_files.keys()) | set(jsonl_files.keys())
        
        for stem in all_stems:
            has_md = stem in md_files
            has_jsonl = stem in jsonl_files
            
            if has_md and has_jsonl:
                md_path = md_files[stem]
                jsonl_path = jsonl_files[stem]
                
                try:
                    # Read markdown content
                    with open(md_path, 'r', encoding='utf-8') as f:
                        md_content = f.read()
                    
                    # Read JSON object
                    with gzip.open(jsonl_path, 'rt', encoding='utf-8') as f:
                        json_obj = json.loads(f.read().strip())
                    
                    # Update text field
                    json_obj['text'] = md_content
                    
                    # Write back
                    with gzip.open(jsonl_path, 'wt', encoding='utf-8') as f:
                        f.write(json.dumps(json_obj))
                    
                    # Delete markdown file
                    md_path.unlink()
                    
                    print(f"Processed: {stem} in {root}")
                    
                except Exception as e:
                    print(f"Error processing {stem} in {root}: {e}")
            
            elif has_md and not has_jsonl:
                print(f"Warning: Orphaned .md file: {md_files[stem]}")
            
            elif has_jsonl and not has_md:
                print(f"Warning: Orphaned .jsonl.gz file: {jsonl_files[stem]}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python merge_mds_into_dolmadocs.py <documents_directory>")
        sys.exit(1)
    
    process_folder(sys.argv[1])