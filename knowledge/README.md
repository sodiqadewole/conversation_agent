# Local knowledge

Add trusted Markdown (`.md`), plain-text (`.txt`), or PDF (`.pdf`) files to this
directory. The conversational agent can search them with its local knowledge
tool.

The lookup is read-only and does not follow symbolic links. Markdown and text
files larger than 1 MB, PDFs larger than 20 MB, and unsupported file types are
ignored. PDFs are indexed page by page and must contain extractable text;
scanned image-only PDFs require OCR before they can be indexed.

The application parses files into heading-aware, word-counted passages and
stores a local SQLite full-text index in `.knowledge_index.sqlite3`. The index
refreshes automatically when files are added, edited, or removed. It also
rebuilds automatically when the chunking or parser configuration changes.
The Streamlit app polls for changes every 20 seconds while a session is open,
and every search performs an immediate synchronization.
