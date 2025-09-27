import os
import math
from typing import List, Tuple

import torch
from PIL import Image
from io import BytesIO
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


DEFAULT_MODEL_NAME = "llamaindex/vdr-2b-multi-v1"


def _select_device() -> Tuple[str, torch.dtype]:
    # Allow override via env var DEVICE: cuda, mps, cpu
    device_env = os.getenv("DEVICE", "").lower()
    if device_env in {"cuda", "cuda:0", "gpu"} and torch.cuda.is_available():
        return ("cuda:0", torch.bfloat16)
    if device_env == "mps" and torch.backends.mps.is_available():
        # Use float16 on MPS by default; allow override via VDR_MPS_DTYPE=fp32
        mps_dtype_env = os.getenv("VDR_MPS_DTYPE", "fp16").strip().lower()
        mps_dtype = torch.float32 if mps_dtype_env in {"fp32", "float32"} else torch.float16
        return ("mps", mps_dtype)
    # Auto-detect
    if torch.cuda.is_available():
        return ("cuda:0", torch.bfloat16)
    if torch.backends.mps.is_available():
        mps_dtype_env = os.getenv("VDR_MPS_DTYPE", "fp16").strip().lower()
        mps_dtype = torch.float32 if mps_dtype_env in {"fp32", "float32"} else torch.float16
        return ("mps", mps_dtype)
    # CPU fallback
    # bfloat16 on CPU is not always supported; use float32 for safety
    return ("cpu", torch.float32)


class VDRMultiEmbedder:
    """
    Wrapper around llamaindex/vdr-2b-multi-v1 to produce query and image embeddings.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_image_tokens: int = 768, use_fast: bool | None = None) -> None:
        self.model_name = model_name
        self.device, self.dtype = _select_device()
        # Initialize HuggingFaceEmbedding via LlamaIndex (stable on MPS)
        target_device = "mps" if self.device == "mps" else ("cuda" if self.device.startswith("cuda") else "cpu")
        self._hf = HuggingFaceEmbedding(
            model_name=self.model_name,
            device=target_device,
            trust_remote_code=True,
        )
        print(f"VDR embedder (LlamaIndex) -> model={self.model_name}, device={target_device}")

    # ------- public API -------
    @torch.inference_mode()
    def encode_queries(self, queries: List[str], dimension: int = 1536) -> torch.Tensor:
        if len(queries) == 0:
            return torch.empty((0, dimension), dtype=self.dtype)
        vecs = []
        for q in queries:
            # Use text embedding for pure-text queries to avoid image-token prompts
            v = self._hf.get_text_embedding(q)
            vecs.append(v[:dimension])
        emb = torch.tensor(vecs, dtype=torch.float32)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb

    @torch.inference_mode()
    def encode_images(self, images: List[Image.Image], dimension: int = 1536) -> torch.Tensor:
        if len(images) == 0:
            return torch.empty((0, dimension), dtype=self.dtype)
        vecs = []
        for img in images:
            with BytesIO() as buf:
                img.save(buf, format="PNG")
                buf.seek(0)
                v = self._hf.get_image_embedding(buf.getvalue())
                vecs.append(v[:dimension])
        emb = torch.tensor(vecs, dtype=torch.float32)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb
