# Chatbot Source PDFs

This directory is the optional source corpus used by the chatbot rebuild pipeline.

Runtime note:
- The deployed app can run from prebuilt index files in backend/app/chatbot/storage.
- Rebuild endpoint (/api/v1/chatbot/rebuild) requires one or more .pdf files here.

If you want to rebuild index:
1. Place PDF files in this folder.
2. Call /api/v1/chatbot/rebuild.
