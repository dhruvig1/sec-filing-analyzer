# src/remote_embed.py
"""Embed the chunks on a faster machine (a free Colab GPU), then build the FAISS index locally.

  1. python src/remote_embed.py export                 -> results/chunks.jsonl
  2. On Colab (T4 GPU): upload chunks.jsonl, run the printed cell -> vectors.npy
  3. python src/remote_embed.py import ~/Downloads/vectors.npy
        -> embeddings/faiss_index__qwen3-embedding-0_6b

Only raw vectors cross between machines; the index is built here with your own packages.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import DEFAULT_EMBEDDING_MODEL, RESULTS_DIR, index_dir_for  # noqa: E402

CHUNKS_FILE = RESULTS_DIR / "chunks.jsonl"

COLAB_CELL = '''
# Runtime > Change runtime type > T4 GPU, then upload chunks.jsonl (Files panel or the prompt below)
!pip -q install -U sentence-transformers
import json, os, numpy as np
from google.colab import files
if not os.path.exists("chunks.jsonl"):
    files.upload()
from sentence_transformers import SentenceTransformer
texts = [json.loads(line)["text"] for line in open("chunks.jsonl")]
model = SentenceTransformer("{model}", device="cuda").half()
vecs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
np.save("vectors.npy", vecs.astype("float32"))
print(vecs.shape)   # should be ({n}, 1024)
files.download("vectors.npy")
'''


def export_chunks(path=CHUNKS_FILE):
    import ingest

    splits = ingest.split_docs(ingest.load_and_chunk_filings())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in splits:
            f.write(json.dumps({"text": d.page_content, "metadata": d.metadata}) + "\n")
    return path, len(splits)


class _NoQueryEmbeddings:
    """Placeholder while building: every vector is precomputed, so nothing gets embedded here."""
    def embed_documents(self, texts):
        raise RuntimeError("vectors are precomputed")

    def embed_query(self, text):
        raise RuntimeError("vectors are precomputed")

    def __call__(self, text):
        raise RuntimeError("vectors are precomputed")


def import_vectors(vectors_path, model, chunks_path=CHUNKS_FILE, index_dir=None):
    import numpy as np
    from langchain_community.vectorstores import FAISS

    rows = [json.loads(line) for line in open(chunks_path)]
    vecs = np.load(vectors_path).astype("float32")
    if vecs.shape[0] != len(rows):
        raise SystemExit(f"{vectors_path} has {vecs.shape[0]} vectors but {chunks_path} has {len(rows)} chunks. "
                         f"Re-export and re-embed so they match.")
    if not np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-2):
        raise SystemExit("Vectors aren't unit length; embed with normalize_embeddings=True.")

    index_dir = Path(index_dir or index_dir_for(model))
    vs = FAISS.from_embeddings(list(zip([r["text"] for r in rows], vecs)), _NoQueryEmbeddings(),
                               metadatas=[r["metadata"] for r in rows])
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    return index_dir, vecs.shape


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    i = sub.add_parser("import")
    i.add_argument("vectors")
    i.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    args = p.parse_args()

    if args.model == DEFAULT_EMBEDDING_MODEL:
        raise SystemExit("This is for the Qwen index; the default MiniLM index is built by ingest.py.")
    if args.cmd == "export":
        path, n = export_chunks()
        print(f"Wrote {n} chunks to {path}\n\nPaste this into a Colab cell:\n" +
              COLAB_CELL.format(model=args.model, n=n))
    else:
        index_dir, shape = import_vectors(Path(args.vectors).expanduser(), args.model)
        print(f"Built index from {shape[0]} vectors (dim {shape[1]}) -> {index_dir}\n"
              f"Next: EMBEDDING_MODEL={args.model} python src/eval_passages.py run")


if __name__ == "__main__":
    main()
