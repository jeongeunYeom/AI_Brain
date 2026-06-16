def chunk_pages(
    pages: list[dict[str, object]],
    filename: str,
    digest: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for page in pages:
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunk_id = f"{digest}:p{page['page']}:c{chunk_index}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "document_id": digest,
                    "document": filename,
                    "page": page["page"],
                    "chunk_index": chunk_index,
                },
            })
            if end == len(text):
                break
            start = max(0, end - chunk_overlap)
            chunk_index += 1
    return chunks
