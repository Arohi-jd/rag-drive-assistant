# RAG Drive Assistant

RAG Drive Assistant is a FastAPI-based retrieval-augmented generation system for Google Drive content. It can sync supported files from Drive, process and chunk document text, generate embeddings, store them in FAISS, and answer questions with Groq using only indexed document context.

The project includes:

- Google Drive sync for PDFs, Google Docs, and TXT files
- Incremental sync with change tracking
- Sliding-window chunking with configurable overlap
- Embedding cache for repeated chunk reuse
- FAISS-based semantic retrieval
- Metadata-aware filtered question answering
- Async API and indexing pipeline
- A static browser UI served from `/`

## Architecture

```text
+------------------+
|   Google Drive   |
+------------------+
          |
          v
+------------------+
|    Processing    |
| clean + chunk    |
+------------------+
          |
          v
+------------------+
|    Embedding     |
| sentence model   |
| + cache          |
+------------------+
          |
          v
+------------------+
|      FAISS       |
| vector index     |
| + metadata store |
+------------------+
          |
          v
+------------------+
|       Groq       |
| grounded answer  |
+------------------+
          |
          v
+------------------+
|      Answer      |
+------------------+
```

## How It Works

1. Files are downloaded from Google Drive or ingested from local disk.
2. Text is extracted from supported formats.
3. The text is cleaned and chunked into overlapping windows.
4. Each chunk is embedded with `sentence-transformers`.
5. Embeddings and chunk metadata are stored in FAISS and `metadata.json`.
6. User queries are embedded and matched against indexed chunks.
7. Top matching chunks are sent to Groq with a grounded prompt.
8. The API returns the answer plus source document names.

## Project Structure

```text
rag-drive-assistant/
├── api/
│   ├── __init__.py
│   └── routes.py
├── connectors/
│   ├── __init__.py
│   └── gdrive.py
├── embedding/
│   ├── __init__.py
│   └── embedder.py
├── processing/
│   ├── __init__.py
│   └── chunker.py
├── search/
│   ├── __init__.py
│   └── faiss_store.py
├── static/
│   └── index.html
├── data/
├── main.py
├── requirements.txt
├── .env.example
├── credentials.json          # you provide this
├── token.json                # generated after first auth
├── synced_files.json         # generated after Drive sync
├── embedding_cache.json      # generated after embeddings
├── faiss_index.bin           # generated index
├── metadata.json             # generated metadata
└── README.md
```

## Setup

### Prerequisites

- Python 3.11 or newer
- A Groq account and API key
- A Google Cloud project with Google Drive API enabled

### Step-by-Step Installation

1. Clone the repository and move into it.

```bash
git clone <your-repo-url>
cd rag-drive-assistant
```

2. Create and activate a virtual environment.

```bash
python3.11 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Create your environment file.

```bash
cp .env.example .env
```

5. Add your Groq API key to `.env`.

6. Download your Google OAuth credentials as `credentials.json` into the project root.

7. Start the server.

```bash
python main.py
```

8. Open the app.

- UI: `http://localhost:8000/`
- Docs: `http://localhost:8000/docs`
- API health: `http://localhost:8000/api/health`

## How To Get Google Drive Credentials

1. Open [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a project.
3. Enable the Google Drive API for that project.
4. Go to `APIs & Services` -> `Credentials`.
5. Click `Create Credentials` -> `OAuth client ID`.
6. If prompted, configure the OAuth consent screen first.
7. Choose `Desktop app` as the application type.
8. Download the JSON credentials file.
9. Rename it to `credentials.json` and place it in the project root.

Notes:

- The first Drive-authenticated request opens a local browser sign-in flow.
- The resulting access token is stored in `token.json`.

## How To Get a Groq API Key

1. Sign in at [Groq Console](https://console.groq.com/).
2. Navigate to the API keys section.
3. Create a new API key.
4. Copy it into your `.env` file:

```env
GROQ_API_KEY=your_actual_key_here
```

## Environment Variables

These are defined in `.env.example`.

| Variable | Required | Description |
|---|---:|---|
| `HOST` | No | Host interface for FastAPI, usually `0.0.0.0` |
| `PORT` | No | Port for the API and UI, default `8000` |
| `GROQ_API_KEY` | Yes for `/api/ask` and `/api/augment` | Groq API key for answer generation |
| `GROQ_MODEL` | No | Groq model name used for completions |
| `GOOGLE_APPLICATION_CREDENTIALS` | Informational | Path to `credentials.json`; the app expects the file in project root |
| `CHUNK_SIZE` | No | Chunk window size in characters |
| `CHUNK_OVERLAP` | No | Overlap between adjacent chunks |
| `EMBEDDING_MODEL` | Informational | Intended embedding model name; current embedder defaults to `all-MiniLM-L6-v2` |
| `FAISS_INDEX_FILE` | Informational | Intended FAISS index file path; current store defaults to `faiss_index.bin` |
| `FAISS_METADATA_FILE` | Informational | Intended metadata file path; current store defaults to `metadata.json` |

Example `.env`:

```env
HOST=0.0.0.0
PORT=8000
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
GOOGLE_APPLICATION_CREDENTIALS=credentials.json
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_FILE=faiss_index.bin
FAISS_METADATA_FILE=metadata.json
```

## Supported File Types

- PDF files, downloaded and later read with `fitz` / PyMuPDF
- Google Docs, exported as plain text during download
- Plain text files (`.txt`)

## API Response Format

All application API endpoints under `/api` return a consistent JSON envelope:

```json
{
  "success": true,
  "message": "Human-readable status message",
  "data": {}
}
```

Error example:

```json
{
  "success": false,
  "message": "No chunks matched the provided metadata filters.",
  "data": {
    "file_name": "quarterly-report.txt",
    "source": null
  }
}
```

## API Endpoints

### `GET /api/health`

Returns service health and indexed counts.

Example response:

```json
{
  "success": true,
  "message": "Server is healthy",
  "data": {
    "server_status": "healthy",
    "documents_indexed": 12,
    "chunks_indexed": 184
  }
}
```

### `POST /api/search`

Semantic search over the indexed chunks.

Example request:

```json
{
  "query": "What does the onboarding checklist require?",
  "top_k": 5
}
```

Example response:

```json
{
  "success": true,
  "message": "Search completed successfully",
  "data": {
    "query": "What does the onboarding checklist require?",
    "results": [
      {
        "text": "New hires must complete security training within 7 days...",
        "similarity": 0.81,
        "metadata": {
          "doc_id": "7ca1f08e7c9796c2",
          "file_name": "onboarding.txt",
          "chunk_index": 0,
          "total_chunks": 4,
          "source": "./data/onboarding.txt",
          "char_count": 912,
          "text": "New hires must complete security training within 7 days..."
        }
      }
    ],
    "total_results": 1
  }
}
```

### `POST /api/ingest`

Ingest a local file from disk into the index.

Example request:

```json
{
  "file_path": "./data/employee-handbook.pdf"
}
```

Example response:

```json
{
  "success": true,
  "message": "Successfully ingested 11 chunks",
  "data": {
    "file_path": "./data/employee-handbook.pdf",
    "chunks_created": 11,
    "index_overview": {
      "documents_indexed": 3,
      "chunks_indexed": 29
    }
  }
}
```

### `POST /api/download-from-drive`

Download named files from Drive and queue them for ingestion.

Example request:

```json
{
  "file_names": [
    "employee-handbook.pdf",
    "security-policy.txt"
  ]
}
```

Example response:

```json
{
  "success": true,
  "message": "Downloaded 2 files. Ingestion in progress...",
  "data": {
    "downloaded_files": [
      "./data/employee-handbook.pdf",
      "./data/security-policy.txt"
    ],
    "requested_files": [
      "employee-handbook.pdf",
      "security-policy.txt"
    ]
  }
}
```

### `POST /api/sync-drive`

Incrementally sync new or modified Drive files.

Example request:

```json
{
  "max_results": 100
}
```

Example response:

```json
{
  "success": true,
  "message": "Google Drive sync completed",
  "data": {
    "new_files": 2,
    "modified_files": 1,
    "skipped_files": 7,
    "failed_files": 0,
    "synced_files": [
      {
        "file_id": "1AbCdEf",
        "file_name": "team-notes.txt",
        "file_path": "./data/team-notes.txt",
        "modified_time": "2026-04-27T10:15:00.000Z",
        "status": "new"
      }
    ],
    "failed_details": []
  }
}
```

### `POST /api/augment`

Generate an answer from the top retrieved chunks without metadata filters.

Example request:

```json
{
  "query": "Summarize the device replacement policy",
  "top_k": 5
}
```

Example response:

```json
{
  "success": true,
  "message": "Augmented response generated successfully",
  "data": {
    "query": "Summarize the device replacement policy",
    "context": [
      "Employees can request a replacement laptop every 36 months...",
      "Managers must approve exceptions for damaged hardware..."
    ],
    "answer": "The device replacement policy comes from device-policy.pdf. Standard laptop replacement is available every 36 months, and manager approval is required for exception cases such as damage.",
    "sources": [
      "device-policy.pdf"
    ]
  }
}
```

### `POST /api/ask`

Generate an answer with optional metadata filtering.

Request fields:

- `query`: required
- `file_name`: optional
- `source`: optional
- `top_k`: optional, default `5`, maximum `20`

Example request:

```json
{
  "query": "When do reimbursements expire?",
  "file_name": "finance-guide.txt",
  "source": "./data/finance-guide.txt",
  "top_k": 5
}
```

Example response:

```json
{
  "success": true,
  "message": "Answer generated successfully",
  "data": {
    "query": "When do reimbursements expire?",
    "filters": {
      "file_name": "finance-guide.txt",
      "source": "./data/finance-guide.txt",
      "top_k": 5
    },
    "matched_chunks": 3,
    "context": [
      "Expense reimbursements must be submitted within 45 days...",
      "Claims older than 45 days require finance approval..."
    ],
    "answer": "According to finance-guide.txt, reimbursements should be submitted within 45 days. Older claims require finance approval.",
    "sources": [
      "finance-guide.txt"
    ]
  }
}
```

No-match example:

```json
{
  "success": false,
  "message": "No chunks matched the provided metadata filters.",
  "data": {
    "file_name": "missing.txt",
    "source": null
  }
}
```

### `GET /api/index-stats`

Return FAISS statistics.

Example response:

```json
{
  "success": true,
  "message": "Index statistics retrieved successfully",
  "data": {
    "total_documents": 184,
    "embedding_dimension": 384,
    "index_size": 184,
    "files_on_disk": 245760
  }
}
```

### `POST /api/clear-index`

Clear the FAISS index and metadata in memory and on disk.

Example response:

```json
{
  "success": true,
  "message": "FAISS index cleared successfully",
  "data": {
    "documents_indexed": 0,
    "chunks_indexed": 0
  }
}
```

## Sample Queries

These are representative examples showing the style of answer the system should produce.

### 1. Policy summary

Query:

```text
What is the password rotation policy?
```

Example answer:

```text
According to security-policy.txt, passwords must be rotated every 90 days and reused passwords are not allowed.
```

### 2. Deadline lookup

Query:

```text
When should travel expenses be submitted?
```

Example answer:

```text
According to finance-guide.txt, travel expenses should be submitted within 30 days of the trip end date.
```

### 3. Process question

Query:

```text
How does new employee onboarding begin?
```

Example answer:

```text
According to onboarding.txt, onboarding begins with HR document completion, account provisioning, and mandatory security training.
```

### 4. Filtered question

Query:

```text
What does the handbook say about remote work?
```

Example answer:

```text
According to employee-handbook.pdf, remote work is allowed with manager approval and employees are expected to maintain core collaboration hours.
```

### 5. Missing information

Query:

```text
What is the company stock vesting schedule?
```

Example answer:

```text
I could not find this information in the documents.
```

## Exceptional Features Implemented

### 1. Incremental Sync

The Drive connector keeps a `synced_files.json` ledger in the project root. It stores:

- Google Drive file ID
- last modified timestamp

During `/api/sync-drive`, the app:

- skips unchanged files
- downloads only new or modified files
- updates the sync ledger after successful processing
- reports synced, skipped, and failed counts

### 2. Embedding Cache

The embedder stores embeddings in `embedding_cache.json` using:

- exact chunk text
- `hashlib.md5(text.encode("utf-8"))` as the cache key

This reduces repeated embedding cost for:

- re-syncs of unchanged chunk content
- repeated document replacement flows
- index rebuild scenarios

### 3. Metadata Filtering

`POST /api/ask` supports:

- `file_name`
- `source`
- `top_k`

This lets users narrow retrieval to a specific document or origin before answer generation. If the filters remove all chunks, the API returns a clear not-found style error instead of generating a weak answer.

### 4. Async Pipeline

The application uses an async architecture for the heavy I/O portions:

- async file reads and writes with `aiofiles`
- concurrent Drive sync processing with `asyncio.gather()`
- FAISS operations wrapped in executors so they do not block the event loop
- async API endpoints for ingestion, sync, search, and answer generation flow

This keeps the server responsive while indexing or syncing larger batches.

## Storage and Generated Files

The app creates and updates these files as it runs:

- `data/` for downloaded source files
- `token.json` for Google OAuth access
- `synced_files.json` for incremental sync state
- `embedding_cache.json` for embedding reuse
- `faiss_index.bin` for vector storage
- `metadata.json` for chunk metadata

## Running the UI

The FastAPI app serves `static/index.html` at:

```text
GET /
```

The UI includes:

- health status indicator
- Drive sync control
- metadata filters
- chat interface backed by `/api/ask`

## Troubleshooting

### Google Drive auth errors

- Confirm `credentials.json` is present in project root
- Delete `token.json` and re-authenticate if needed
- Verify Drive API is enabled in Google Cloud

### Groq errors

- Verify `GROQ_API_KEY` is set in `.env`
- Confirm the configured model is available in your Groq account

### Empty answers

- Check `/api/health` and `/api/index-stats`
- Make sure documents have been ingested or synced
- Verify metadata filters are not too restrictive

### FAISS issues

- Delete `faiss_index.bin` and `metadata.json` to rebuild from scratch
- Re-run ingestion or Drive sync

## License

MIT License
