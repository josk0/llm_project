Here are scripts to build a dataset from a zotero liberary

1. You need to get a dump of your PDFs (eg using https://github.com/josk0/zoteromd, still private though)
2. optional: remove annotations from PDF
3. run OCR, on cluster using ocr.sub (adjust paths in conda_wrapper_ocr.sh)
4. delete orphaned files using prune_dolma_dataset.py
5. run additional markdown cleaning using clean_mds_in_folder.sh
6. inspect the results, then run merge_mds_into_dolmadocs.py.
7. Make sure no files or folder have semicolons in the name
7. run dolma taggers 
8. output the dataset with `dolma -c zotero-mixer.json mix`

for dolma taggers
```bash
dolma tag \
	--documents "v0.61/documents/**/*.jsonl.gz" \
 	--taggers ft_lang_id_en_paragraph_v2 \
 			  random_number_v1 \
 	--processes 8
```

Requirements
- dolma
- markdowncleaner
- tqdm