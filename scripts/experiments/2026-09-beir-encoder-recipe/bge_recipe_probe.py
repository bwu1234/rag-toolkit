"""Which document recipe produced Pyserini's BGE Faiss vectors? (public benchmarks plan, Phase 0).

FROZEN RECORD, not maintained tooling. Pyserini documents the query side of
its `bge-base-en-v1.5.faiss` BEIR runs (the 2CR search command) but not the
command that encoded the prebuilt 2024-01-07 document indexes, and its NFCorpus
tutorial encodes with `--pooling mean`, which is not BGE's intended pooling.
The model name alone does not identify the configuration, so recover it from
the vectors: re-encode sample SciFact documents under each candidate recipe
and compare with the stored vector for the same docid. Samples cover documents
under 256 tokens, between 256 and 512, and over 512, so the truncation length
is distinguishable.

Result (2026-09-29, transformers 5.14.1, torch 2.13.0, CPU, model revision
a5beb1e3e68b9ab74eb54cfd186867f64f240e1a): CLS pooling, L2 normalisation,
512-token truncation over title + whitespace + text matches every sample at
cosine >= 0.999999. Every other combination falls to <= 0.975 on at least
one sample. The BERT tokenizer drops the separator, so " " and "\\n" are
indistinguishable here; see docs/beir-reference-protocol.md.

Needs the unpacked archive and index (neither is committed):
    python scripts/experiments/2026-09-beir-encoder-recipe/bge_recipe_probe.py \\
        path/to/scifact path/to/faiss-flat.beir-v1.0.0-scifact.bge-base-en-v1.5.20240107
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-base-en-v1.5"


def read_flat_index(path: Path) -> np.ndarray:
    """Read a Faiss IndexFlat file without Faiss: header, then raw float32 codes."""
    raw = path.read_bytes()
    assert raw[:4] in (b"IxFI", b"IxF2"), raw[:4]
    (d,) = struct.unpack_from("<i", raw, 4)
    (n,) = struct.unpack_from("<q", raw, 8)
    # 8+8 dummy bytes, is_trained (1 byte), metric_type (int32, 0 = inner product)
    (metric,) = struct.unpack_from("<i", raw, 33)
    (size,) = struct.unpack_from("<q", raw, 37)
    assert metric == 0 and size == n * d
    return np.frombuffer(raw, dtype="<f4", count=size, offset=45).reshape(n, d)


def main() -> None:
    corpus_dir, index_dir = Path(sys.argv[1]), Path(sys.argv[2])
    vecs = read_flat_index(index_dir / "index")
    ids = (index_dir / "docid").read_text().split()
    row = {doc_id: i for i, doc_id in enumerate(ids)}
    docs = {}
    for line in (corpus_dir / "corpus.jsonl").open(encoding="utf-8"):
        o = json.loads(line)
        docs[o["_id"]] = o
    print(f"{len(ids)} vectors of dim {vecs.shape[1]}; docid order == corpus order: {ids == list(docs)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    lens = {i: len(tok(f'{docs[i]["title"]} {docs[i]["text"]}')["input_ids"]) for i in ids[:400]}
    sample = (
        [i for i in ids[:400] if lens[i] < 200][:3]
        + [i for i in ids[:400] if 300 < lens[i] < 500][:3]
        + [i for i in ids[:400] if lens[i] > 600][:2]
    )
    print("sample token lengths", [lens[i] for i in sample])

    recipes = {
        "title + ' ' + text": lambda o: f'{o["title"]} {o["text"]}',
        "title + '\\n' + text": lambda o: f'{o["title"]}\n{o["text"]}',
        "text only": lambda o: o["text"],
    }

    @torch.no_grad()
    def encode(text: str, pooling: str, max_len: int) -> np.ndarray:
        batch = tok([text], max_length=max_len, truncation=True, return_tensors="pt")
        hidden = model(**batch)[0]
        mask = batch["attention_mask"][..., None]
        e = hidden[:, 0] if pooling == "cls" else (hidden * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(e, dim=-1)[0].numpy()

    for name, recipe in recipes.items():
        for pooling in ("cls", "mean"):
            for max_len in (256, 512):
                cos = [float(encode(recipe(docs[i]), pooling, max_len) @ vecs[row[i]]) for i in sample]
                print(f"{name:22} {pooling:4} {max_len}: min {min(cos):.6f}  " + " ".join(f"{c:.4f}" for c in cos))


if __name__ == "__main__":
    main()
