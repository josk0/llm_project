"""
Preprocess raw markdown documents into token-bounded chunks for Qwen finetuning.

Splits academic documents into paragraphs while respecting token limits.
Avoids splitting inside protected blocks (code fences, equations).
Outputs preprocessed chunks ready for instruction tuning.

Chunking strategy:
  - Parse documents into paragraphs (separated by blank lines)
  - Preserve content inside protected blocks (```...```, $$...$$, \\[...\\])
  - Greedily fill chunks up to target context token budget
  - Handle edge cases: tiny sections merged, huge sections split
  - Create context-response pairs from consecutive chunks
"""

import os
import json
import gzip
import logging
from pathlib import Path
from typing import List, Tuple
from transformers import AutoTokenizer
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class MarkdownChunker:
    """
    Chunks academic markdown documents into context-response pairs
    respecting token budgets and semantic boundaries.
    """

    def __init__(self,
                 tokenizer_name: str,
                 target_context_tokens: int = 1200,
                 target_response_tokens: int = 200,
                 max_total_tokens: int = 1700):
        """
        Initialize the chunker with token budgets.

        Args:
            tokenizer_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-3B-Instruct")
            target_context_tokens: Ideal context length (will chunk when exceeded)
            target_response_tokens: Ideal response length (continuation paragraph)
            max_total_tokens: Hard limit for total example length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.target_context = target_context_tokens
        self.target_response = target_response_tokens
        self.max_total = max_total_tokens

        # Thresholds for section size edge cases
        self.tiny_section_threshold = 300
        self.huge_section_threshold = 2000

    def _parse_paragraphs(self, text: str) -> List[str]:
        """
        Split text into logical units: sections, paragraphs, preserving structure.

        Returns list of text blocks (sections, subsections, paragraphs).
        """
        lines = text.split('\n')
        blocks = []
        current_block = []
        in_protected = False

        for line in lines:
            # Track protected blocks (code fences, equations)
            if line.strip().startswith('```'):
                in_protected = not in_protected
            elif line.strip().startswith('$$'):
                in_protected = not in_protected
            elif line.strip().startswith('\\['):
                in_protected = not in_protected

            # Paragraph boundary: blank line (unless in protected block)
            if not line.strip() and not in_protected:
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
            else:
                current_block.append(line)

        if current_block:
            blocks.append('\n'.join(current_block))

        return [b for b in blocks if b.strip()]  # Filter empty

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _chunk_with_budget(self, text: str) -> List[str]:
        """
        Greedily chunk text respecting token budget.

        Returns list of chunks, each under target_context tokens.
        """
        paragraphs = self._parse_paragraphs(text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            # Handle tiny sections: merge with next
            if para_tokens < self.tiny_section_threshold and current_chunk:
                current_chunk.append(para)
                current_tokens += para_tokens
                continue

            # Handle huge sections: split at paragraph level
            if para_tokens > self.huge_section_threshold:
                # If we have accumulated chunk, save it first
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split huge paragraph further (simple: by sentences)
                sub_paras = para.split('. ')
                sub_chunk = []
                sub_tokens = 0

                for sub_para in sub_paras:
                    sub_para = sub_para + '.' if not sub_para.endswith('.') else sub_para
                    sub_tokens_count = self._count_tokens(sub_para)

                    if sub_tokens + sub_tokens_count > self.target_context:
                        if sub_chunk:
                            chunks.append(' '.join(sub_chunk))
                        sub_chunk = [sub_para]
                        sub_tokens = sub_tokens_count
                    else:
                        sub_chunk.append(sub_para)
                        sub_tokens += sub_tokens_count

                if sub_chunk:
                    chunks.append(' '.join(sub_chunk))
                continue

            # Normal case: greedy fill
            if current_tokens + para_tokens > self.target_context and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def chunk_document(self, text: str) -> List[Tuple[str, str]]:
        """
        Chunk document and create context-response pairs.

        For each chunk, use it as context and append continuation as response.

        Returns list of (context, response) tuples.
        """
        chunks = self._chunk_with_budget(text)
        pairs = []

        # Create context-response pairs: each chunk → next chunk as response
        for i in range(len(chunks) - 1):
            context = chunks[i]
            response = chunks[i + 1]

            context_tokens = self._count_tokens(context)
            response_tokens = self._count_tokens(response)
            total_tokens = context_tokens + response_tokens

            # Enforce max total length
            if total_tokens > self.max_total:
                # Truncate response if needed
                if response_tokens > self.target_response:
                    # Truncate response to fit
                    tokens = self.tokenizer.encode(response, add_special_tokens=False)
                    tokens = tokens[:self.max_total - context_tokens]
                    response = self.tokenizer.decode(tokens, skip_special_tokens=True)
                else:
                    # Skip this pair if context alone is too long
                    continue

            pairs.append((context, response))

        return pairs


def preprocess_files(
    data_path: str,
    output_dir: str,
    tokenizer_name: str,
    target_context_tokens: int = 1200,
    target_response_tokens: int = 200,
    train_split: float = 0.8,
    seed: int = 42,
) -> None:
    """
    Preprocess all gzipped JSONL files in data_path.

    Args:
        data_path: Directory with gzipped JSONL files
        output_dir: Output directory for preprocessed data
        tokenizer_name: HuggingFace tokenizer
        target_context_tokens: Context token budget
        target_response_tokens: Response token budget
        train_split: Fraction for training (rest for eval)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    chunker = MarkdownChunker(
        tokenizer_name,
        target_context_tokens=target_context_tokens,
        target_response_tokens=target_response_tokens,
    )

    # Collect all examples
    all_examples = []
    doc_count = 0
    example_count = 0

    logging.info(f"Starting preprocessing from {data_path}")

    for filename in sorted(os.listdir(data_path)):
        if not filename.endswith('.gz'):
            continue

        filepath = os.path.join(data_path, filename)
        logging.info(f"Processing {filename}")

        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        doc = json.loads(line)
                        if 'text' not in doc or not doc['text']:
                            continue

                        text = doc['text']

                        # Chunk document
                        pairs = chunker.chunk_document(text)

                        for context, response in pairs:
                            all_examples.append({
                                "context": context,
                                "response": response,
                                "source_doc": filename,
                                "source_line": line_num,
                            })
                            example_count += 1

                        doc_count += 1

                        if doc_count % 100 == 0:
                            logging.info(f"  Processed {doc_count} documents, {example_count} examples")

                    except json.JSONDecodeError as e:
                        logging.warning(f"  Line {line_num}: JSON decode error: {e}")
                        continue

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            continue

    logging.info(f"Total: {doc_count} documents, {example_count} examples")

    # Shuffle and split
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * train_split)
    train_examples = all_examples[:split_idx]
    eval_examples = all_examples[split_idx:]

    logging.info(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write train split
    train_path = os.path.join(output_dir, "train.jsonl.gz")
    with gzip.open(train_path, 'wt', encoding='utf-8') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')
    logging.info(f"Wrote train data to {train_path}")

    # Write eval split
    eval_path = os.path.join(output_dir, "eval.jsonl.gz")
    with gzip.open(eval_path, 'wt', encoding='utf-8') as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + '\n')
    logging.info(f"Wrote eval data to {eval_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess markdown documents for Qwen finetuning")
    parser.add_argument("--data-path", required=True, help="Path to directory with gzipped JSONL files")
    parser.add_argument("--output-dir", required=True, help="Output directory for preprocessed data")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace tokenizer name")
    parser.add_argument("--context-tokens", type=int, default=1200, help="Target context token length")
    parser.add_argument("--response-tokens", type=int, default=200, help="Target response token length")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/eval split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    preprocess_files(
        data_path=args.data_path,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        target_context_tokens=args.context_tokens,
        target_response_tokens=args.response_tokens,
        train_split=args.train_split,
        seed=args.seed,
    )

    logging.info("Preprocessing complete!")
