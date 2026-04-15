# Vector Store — Persistent Storage
This folder is **auto-generated** by the index builder.

## Files
| File | Purpose |
|------|---------|
| `faiss_index.bin` | FAISS flat inner-product index |
| `chunk_metadata.json` | Chunk text + source metadata (file name, page, chunk ID) |

## Rebuild
```bash
cd backend
python -m app.chatbot.ingestion.build_index
```

> **Do not commit** these files to version control — they are large binary blobs
> that can be regenerated from the source PDFs in `chatbot/`.
