# Visual Document Retrieval with vdr-2b-multi-v1 + Pinecone + Streamlit

Search a folder of screenshots using multilingual semantic search without OCR. Images are embedded with `llamaindex/vdr-2b-multi-v1` and stored in Pinecone. A Streamlit app lets you query and view the top-5 matches with a click-to-enlarge preview.

## Features
- Multilingual cross-lingual search (IT, ES, EN, FR, DE)
- No OCR needed: direct image embeddings
- Pinecone vector index with `image_path` metadata
- Streamlit UI with enlarge-on-click
- Matryoshka-ready: reduce dimensions (e.g., 512) with minimal quality loss

## Setup

1. Create and activate a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Configure environment
- Copy `.env.example` to `.env` and fill `PINECONE_API_KEY`.
- Optional: tune `DEVICE` (cuda/mps/cpu) and `DIMENSION` (default 1536).

## Index your screenshots

Put your images under `./screenshots/` or set `SCREENSHOTS_DIR` in `.env`.

Run the indexer:
```
python scripts/index_images.py --folder ./screenshots --index vdr-images --namespace default --batch_size 16 --dimension 1536
```
You can omit flags if you use `.env` variables (see `.env.example`).

This will:
- Read images, embed via vdr-2b-multi-v1
- Create Pinecone index if it does not exist
- Upsert vectors with metadata: `{ image_path, file_name, folder }`

## Run the Streamlit app
```
streamlit run app.py
```
Use the sidebar to adjust index, namespace, and embedding dimension. Enter a query (any of the 5 languages) and view the top-5 results. Click "View full size" to open a modal preview.

## Notes
- GPU recommended for speed; CPU works but is slower.
- Memory usage scales with image resolution (we cap to 768 image tokens internally).
- To reduce storage/query costs, lower `DIMENSION` (e.g., 512). The model supports Matryoshka truncation by taking the first N dimensions.

## Troubleshooting
- Torch or flash attention warnings on CPU/MPS are harmless; we fall back gracefully.
- If images don't display in the app, ensure `image_path` metadata points to accessible files on the same machine.
- Pinecone errors: verify `PINECONE_API_KEY`, cloud/region, and index name.
