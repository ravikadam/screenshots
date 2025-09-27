import argparse
import os
import sys
import hashlib
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Local imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from embedding import VDRMultiEmbedder
from pinecone_helper import get_pinecone_client, ensure_index, upsert_vectors, describe_stats


IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP",
}


def collect_images(folder: Path) -> List[Path]:
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(folder.rglob(f"*{ext}"))
    return sorted(paths)


def id_for_path(p: Path) -> str:
    # Deterministic id from path
    return hashlib.md5(str(p.resolve()).encode("utf-8")).hexdigest()


def sanitize_embeddings(emb: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0, then row-normalize; drop rows that remain all-zero."""
    # Replace non-finite values
    emb = np.where(np.isfinite(emb), emb, 0.0).astype(np.float32)
    # Row norms
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    # Avoid division by zero; build a 1D mask for rows with non-zero norm
    mask = norms[:, 0] > 0
    # Broadcast-safe division (k,1536) / (k,1)
    if np.any(mask):
        emb[mask] = emb[mask] / norms[mask]
    return emb


def log(msg: str, verbose: bool = True):
    if not verbose:
        return
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Index screenshots into Pinecone.")
    parser.add_argument("--folder", type=str, default=os.getenv("SCREENSHOTS_DIR", "./screenshots"))
    parser.add_argument("--index", type=str, default=os.getenv("INDEX_NAME", "vdr-images"))
    parser.add_argument("--namespace", type=str, default=os.getenv("NAMESPACE", "default"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dimension", type=int, default=int(os.getenv("DIMENSION", 1536)))
    parser.add_argument("--verbose", action="store_true", help="Enable detailed per-file logging")

    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    image_paths = collect_images(folder)
    if not image_paths:
        raise SystemExit(f"No images found in {folder} with extensions {sorted(IMAGE_EXTS)}")

    print(f"Found {len(image_paths)} images in {folder}")
    if args.verbose:
        for p in image_paths:
            log(f"Discovered file: {p}", verbose=True)

    # Initialize embedder
    embedder = VDRMultiEmbedder()

    # Pinecone
    pc = get_pinecone_client()
    index = ensure_index(pc, index_name=args.index, dimension=args.dimension)

    # Process in batches
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Indexing"):
        batch_paths = image_paths[i : i + args.batch_size]

        # Load images
        pil_images = []
        ids = []
        metas = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                log(f"Opened: {p} size={img.width}x{img.height}", verbose=args.verbose)
            except Exception as e:
                print(f"Failed to open {p}: {e}")
                continue
            pil_images.append(img)
            ids.append(id_for_path(p))
            metas.append({
                "image_path": str(p.resolve()),
                "file_name": p.name,
                "folder": str(p.parent),
            })

        if not pil_images:
            continue

        log(f"Embedding {len(pil_images)} image(s)...", verbose=args.verbose)
        embeddings = embedder.encode_images(pil_images, dimension=args.dimension)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        embeddings = sanitize_embeddings(embeddings)

        # Report any rows that ended up zero (likely due to NaN cleanup) and skip them
        row_norms = np.linalg.norm(embeddings, axis=1)
        zero_mask = row_norms == 0
        zero_rows = int(np.sum(zero_mask))
        if zero_rows:
            print(f"Note: {zero_rows} embedding(s) had zero norm after sanitization (skipping those).")

        if args.verbose:
            for _id, meta, norm in zip(ids, metas, row_norms):
                log(f"Embedding norm id={_id[:8]} path={meta['image_path']} norm={norm:.6f}", verbose=True)

        if args.verbose:
            for _id, meta in zip(ids, metas):
                log(f"Upserting id={_id[:8]} path={meta['image_path']}", verbose=True)

        # Filter out zero-norm rows before upsert
        keep_mask = ~zero_mask
        ids_keep = [i for i, k in zip(ids, keep_mask) if k]
        metas_keep = [m for m, k in zip(metas, keep_mask) if k]
        emb_keep = embeddings[keep_mask]
        if len(ids_keep) == 0:
            print("All embeddings in this batch are zero after sanitization; skipping batch.")
            continue

        resp = upsert_vectors(index, ids=ids_keep, vectors=emb_keep, metadatas=metas_keep, namespace=args.namespace)
        try:
            upserted = resp.get("upserted_count") if isinstance(resp, dict) else getattr(resp, "upserted_count", None)
        except Exception:
            upserted = None
        print(
            f"Batch {i//args.batch_size + 1}: attempted {len(ids_keep)} upsert(s)"
            + (f", skipped {zero_rows} zero-norm" if zero_rows else "")
            + (f", server acknowledged {upserted}" if upserted is not None else "")
        )

    print("Indexing complete. Fetching index stats...")
    stats = describe_stats(index, namespace=args.namespace)
    print(stats)


if __name__ == "__main__":
    main()
