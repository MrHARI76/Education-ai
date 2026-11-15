# NexusEd AI: Code Walkthrough

This document explains every file, the functions inside it (in the order they appear),
and how data moves through the system. Read the **Glossary** at the end first if a
term is new.

---

## 1. The big picture

```
                ┌──────────────── Upload flow (once per document) ───────────────┐
 Browser ──POST /api/upload──► main.py ──► pipeline.py
                                             │ parser.py      file → sections → chunks
                                             │ embeddings.py  chunks → 384-number vectors
                                             │ vector_store.py vectors → Qdrant
                                             │ keywords.py    full text → key phrases
                                             │ mindmap.py     headings + keywords + excerpts → LLM → map
                                             └ db.py          status + map → SQLite
 Browser ──GET /api/documents/{id} (every 1.5 s)──► status, then the mind map

                ┌──────────────── Click flow (once per concept) ─────────────────┐
 Browser ──POST /api/explain──► main.py ──► rag.py
                                             │ db.py          cached? return it
                                             │ embeddings.py  query → vector
                                             │ vector_store.py top 20 similar chunks
                                             │ embeddings.py  cross-encoder rerank → top 4
                                             │ llm.py         prompt + 4 chunks → JSON explanation
                                             └ db.py          save to cache
```

---

## 2. Backend files (`app/`)

### `config.py`: settings
**Purpose:** one place for every configurable value, read from environment variables or `.env`.

- `Settings` (class): fields such as `ollama_url`, `llm_model`, `qdrant_url`, `max_upload_mb`, each with a default. pydantic-settings fills them from env vars of the same name.
  - `upload_dir` / `db_path` (properties): paths derived from `data_dir`.
- `get_settings()`: builds `Settings` once (`lru_cache`) so every module shares the same object.
- `settings`: the shared instance other modules import.

**Data flow:** environment → `settings` → used by every other module.

### `models.py`: data shapes
**Purpose:** Pydantic classes that describe JSON. FastAPI uses them to validate requests and responses; `llm.py` sends some of them to Ollama as a JSON Schema so the model must answer in that exact shape.

- **LLM outputs:** `LLMChildNode`, `LLMBranch`, `LLMMindMap` (nested mind map), `Quiz`, `LLMExplanation`, `LLMAnswer`.
- **Graph format:** `MindMapNode` (`id, label, summary, parent_id, level`) and `MindMap` (flat list + `generated_by` = `"llm"` or `"fallback"`).
- **API bodies:** `Source`, `ExplainRequest`, `ExplainResponse` (LLM explanation + server-added `sources`, `cached`), `AskRequest`, `AskResponse`, `UploadResponse`, `DocumentSummary`, `DocumentDetail`, `HealthResponse`.

### `db.py`: SQLite storage
**Purpose:** remember documents (status, progress, mind map) and cache explanations in `data/nexus.db`.

- `connect()`: context manager: opens a connection, commits, always closes. A new connection per call is thread-safe.
- `init_db()`: creates folders and the two tables (`documents`, `explanations`).
- `create_document()`: inserts a row with status `queued`.
- `update_status()`: saves the pipeline stage, progress % and optional error.
- `save_mindmap()`: stores the mind map as JSON text.
- `get_document()`: returns one row as a dict (mind map parsed back from JSON).
- `list_documents()`: newest documents first.
- `delete_document()`: removes the row and its cached explanations.
- `get_cached_explanation()` / `cache_explanation()`: read/write the cache keyed by `(doc_id, node_id, level)`.

### `parser.py`: files → sections → chunks
**Purpose:** extract clean text with "where it came from" metadata.

- `ParseError`: exception whose message is shown to the user.
- `Section` (dataclass): `text, source, file_type, location, heading`.
- `clean_text()`: collapses repeated spaces and blank lines.
- `check_extension()`: accepts `.pdf/.docx/.pptx`; gives a "Save As" hint for `.doc/.ppt`.
- `parse_pdf()`: one section per page (`"page N"`), using pypdf.
- `_iter_docx_blocks()`: walks the Word XML so paragraphs and tables come out in document order.
- `_table_to_text()`: a table becomes lines like `Input | Output`.
- `parse_docx()`: paragraphs are grouped under the latest `Title`/`Heading 1-3`; each group is a section (`"section N"`), and headings are collected as structure hints.
- `parse_pptx()`: one section per slide (`"slide N"`): title placeholder = heading, other text boxes = body, speaker notes prefixed `Notes:`.
- `dedupe_keep_order()`: removes duplicate headings, case-insensitive.
- `parse_document()`: picks the right parser, turns crashes into a friendly `ParseError`, and rejects files with under 200 characters ("scanned PDF?").
- `chunk_sections()`: `RecursiveCharacterTextSplitter` (1000 characters, 150 overlap) splits each section; every chunk keeps its section's metadata plus a running `chunk_index`.

### `embeddings.py`: the two local ML models
**Purpose:** turn text into vectors, and score passage relevance.

- `load_models()`: loads `all-MiniLM-L6-v2` (embedder) and `ms-marco-MiniLM-L-6-v2` (cross-encoder) once.
- `embed_texts()`: list of texts → array of normalised 384-number vectors.
- `rerank()`: scores each (query, passage) pair with the cross-encoder and returns the indexes of the best `top_k`.

### `vector_store.py`: Qdrant
**Purpose:** store chunk vectors and find the most similar ones for a document.

- `get_client()`: one Qdrant client (`":memory:"` = in-process Qdrant for tests).
- `ensure_collection()`: creates the collection (384 dims, cosine) and a `doc_id` index; retries for ~30 s because Qdrant may start after the app.
- `is_healthy()`: used by `/api/health`.
- `_doc_filter()`: a filter that matches only one document's points.
- `upsert_chunks()`: saves points (UUID id + vector + payload) in batches of 256.
- `search()`: top-N most similar chunks, always filtered by `doc_id`.
- `delete_document()`: deletes all points of one document.

### `keywords.py`: key concepts
**Purpose:** find the important phrases without any AI model.

- `is_useful_phrase()`: drops numbers and phrases that start/end with a stopword ("of the").
- `dedupe_by_meaning()`: embeds phrases and skips any whose cosine similarity to an already-kept, better-scored phrase is > 0.85 (e.g. "neural network" vs "neural networks").
- `extract_keywords()`: YAKE (up to 3-word phrases, top 40), sort by score (lower = better), filter, dedupe.

### `llm.py`: talking to Ollama
**Purpose:** get validated JSON from the local LLM.

- `LLMError`: the model answered with unusable output. `LLMUnavailable` (subclass): Ollama down, timed out, or model not pulled.
- `strip_fences()`: removes stray Markdown fences like ```` ```json ````.
- `_post_chat()`: POST `/api/chat` with `format = JSON schema`, `options = {num_ctx, temperature}`, `stream: false`; turns network problems into `LLMUnavailable`.
- `parse_reply()`: `json.loads` + Pydantic validation.
- `chat_json()`: tries twice; the second attempt adds "Return only valid JSON matching the schema". Raises `LLMError` if both fail.
- `health()`: `GET /api/tags` → (Ollama reachable?, model pulled?).

### `prompts.py`: all prompt text
**Purpose:** keep every prompt in one readable place.

- `MINDMAP_SYSTEM` / `MINDMAP_USER`: rules: 4–7 branches, 2–5 children, labels ≤ 5 words, summaries ≤ 20 words, use the document's own terms.
- `EXPLAIN_SYSTEM` / `EXPLAIN_USER`: use ONLY the context; say so if it isn't covered; relatable examples for Indian students; exactly 3 key points; 4-option quiz.
- `LEVEL_INSTRUCTIONS`: `standard` vs `simple` ("explain to a 12-year-old").
- `ASK_SYSTEM` / `ASK_USER`: free-form question answering.

### `mindmap.py`: building the map
**Purpose:** produce the flat node list the graph needs, with or without the LLM.

- `pick_excerpts()`: 10 chunks spread evenly through the document (first 400 characters each).
- `build_prompt()`: fills `MINDMAP_USER` with file name, ≤ 60 headings, ≤ 30 keywords and the excerpts, small enough for the context window.
- `short_label()`: trims labels to a few words.
- `flatten()`: nested tree → flat list. Assigns `n0` (centre), `n1`, `n2`… in order, sets `parent_id` and `level`, enforces ≤ 7 branches / ≤ 5 children, drops duplicate labels.
- `build_mindmap()`: asks the LLM (`chat_json` with `LLMMindMap`) and flattens the result.
- `first_sentence_mentioning()`: finds a sentence in the document containing a term (summaries for the fallback map).
- `fallback_mindmap()`: no LLM. Branches are headings (or top keywords if there are fewer than 3 headings). Children are keywords found in that branch's text, then leftovers shared out. **The demo never shows a blank screen.**

### `rag.py`: explaining a concept
**Purpose:** retrieval-augmented explanations and answers.

- `NotFound` / `NotReady`: become HTTP 404 / 409 in `main.py`.
- `_retrieve_sync()`: embed query → Qdrant top 20 → cross-encoder → best 4.
- `retrieve()`: runs `_retrieve_sync` in a worker thread (CPU work must not block the server).
- `format_context()`: joins chunks as `[slide 3] text…`.
- `collect_sources()`: source chips from chunk metadata, deduplicated. **Never generated by the LLM**, so they can't be invented.
- `load_ready_document()`: 404 if missing, 409 if still processing.
- `explain_node()`: cache check → find node and parent → query `"{label}. {parent}. {topic}"` → retrieve → prompt → LLM (temperature 0.4 for "simple" so the rewrite differs) → add sources → cache → return.
- `answer_question()`: same retrieval for a free-form question (`/api/ask`).

### `pipeline.py`: the background job
**Purpose:** run all ingestion steps after the upload response has already been sent.

- `_embed_and_store()`: clears old points (for a retry), embeds all chunks, saves them to Qdrant.
- `process_document()`: `parsing 10 → embedding 35 → keywords 60 → building_map 75 → ready 100`, saving status after each step. If the LLM fails, it uses `fallback_mindmap`. Any error becomes status `failed` with a readable message.

### `main.py`: the web server
**Purpose:** HTTP routes, startup, and serving the frontend.

- `mark_interrupted_documents()`: documents stuck "processing" after a restart are marked failed (so the Retry button appears instead of spinning forever).
- `lifespan()`: on startup: create DB, load ML models once, connect to Qdrant (the app still starts if Qdrant is down, so the UI can show a red chip).
- `require_document()`: fetch or 404.
- `llm_http_error()`: `LLMUnavailable` → 503, other `LLMError` → 502.
- `GET /api/health` → `health()`.
- `save_upload()`: streams the file to disk in 1 MB blocks; 413 if over the limit, 400 if empty.
- `POST /api/upload` → `upload()`: check extension (400), save as `data/uploads/{doc_id}{ext}`, create DB row, schedule `process_document`, return `{doc_id}` immediately.
- `GET /api/documents` → recent list. `GET /api/documents/{id}` → status + mind map.
- `POST /api/documents/{id}/retry`: re-runs the pipeline (Retry button).
- `DELETE /api/documents/{id}`: file + Qdrant points + cache + row.
- `POST /api/explain`, `POST /api/ask`: call `rag.py`, map errors to HTTP codes.
- `app.mount("/", StaticFiles(...))` is **last**, so it doesn't hide the `/api` routes.

---

## 3. Frontend files (`frontend/`)

### `index.html`
The page skeleton: aurora background, top bar (brand, map toolbar, health chips), three `<section>` screens (`screen-upload`, `screen-processing`, `screen-map`), the concept drawer and the toast container. Loads Tailwind, Cytoscape.js and Inter from CDNs, then `styles.css` and `app.js`.

### `styles.css`
Everything Tailwind can't express simply: drifting aurora blobs, glass panels, the animated gradient border on drag-over, stepper pulse/check, drawer (side panel → bottom sheet under 768 px), shimmer skeleton, quiz green/red shake, toasts, and the `prefers-reduced-motion` switch-off.

### `app.js` (sections in order)
1. **Utilities:** `$`, `escapeHtml` (document text is never inserted as raw HTML), `formatBytes`, `toast`, safe `storage`.
2. **API helpers:** `api()` wraps `fetch` and turns FastAPI's `detail` into an `Error`; `API.*` has one function per endpoint.
3. **Screens/state:** `state` object; `showScreen()`; `openDocument()` (map if ready, else processing). The URL gets `#doc=…` so a refresh returns to the same document.
4. **Health chips:** `refreshHealth()` every 15 s: green / amber (model missing) / red, with tooltips.
5. **Upload:** `validateFile()` (instant browser-side checks), `handleFile()`, drag-and-drop handlers, `loadRecent()` and the recent-list click handler (open / delete).
6. **Polling:** `showProcessing()`, `updateProcessing()` (maps status → stepper step), `startPolling()` every 1.5 s until `ready` (→ map) or `failed` (→ error card with Retry).
7. **Graph rendering:** `tint()` lighter child colours; `buildElements()` flat nodes → Cytoscape nodes/edges with colour and size; `GRAPH_STYLE`; `radialPositions()`, our own radial layout where each branch gets a slice of the circle sized by its number of children; `runLayout()`; `introAnimation()` (nodes grow out of the centre); `setupGraphEvents()` (hover highlight, click → drawer); `renderMap()`; `exportPng()`.
8. **Drawer:** `openDrawer()`, `keepNodeVisible()` (pans so the node isn't hidden), `closeDrawer()`, `showSkeleton()`, `loadExplanation()` (ignores late answers if you clicked elsewhere), `renderExplanation()`.
9. **Quiz:** `renderQuiz()`, `setupQuiz()`: correct → green, wrong → red shake + reveal correct + "why".
10. **Progress:** `markExplored()` (green ring + ✓, saved in `localStorage` per document), `restoreExplored()`, `updateProgress()`.
11. **Start-up:** `setupButtons()` (Fit, Re-layout, PNG, New, Retry, Esc, Explain simpler, resize) and `start()`.

---

## 4. Upload flow, step by step

1. User drops `biology.pptx` on the drop zone → `handleFile()` checks extension/size in the browser.
2. `POST /api/upload` (multipart). `main.upload()` checks the extension again (never trust the browser), streams the file to `data/uploads/<doc_id>.pptx`, inserts a `queued` row, schedules `process_document`, and returns `{"doc_id": "..."}` **immediately**.
3. Browser switches to the processing screen and polls `GET /api/documents/<doc_id>` every 1.5 s.
4. In the background, `pipeline.process_document()`:
   1. **parsing (10%)**: `parse_document()` → sections (`slide 1…`) + headings; `chunk_sections()` → ~1000-character chunks with metadata.
   2. **embedding (35%)**: `embed_texts()` → one vector per chunk; `upsert_chunks()` → Qdrant with payload `{doc_id, text, source, file_type, location, heading, chunk_index}`.
   3. **keywords (60%)**: `extract_keywords()` → ~20–40 deduplicated key phrases.
   4. **building_map (75%)**: `build_mindmap()` sends file name + headings + keywords + 10 excerpts to Ollama with the `LLMMindMap` schema → validated tree → `flatten()` → `n0…nN`. On failure → `fallback_mindmap()`.
   5. **ready (100%)**: mind map saved to SQLite.
5. The next poll sees `ready` → `renderMap()` builds the Cytoscape graph and plays the intro animation.

## 5. Click flow, step by step

1. User clicks node `n7` → `openDrawer()` slides the drawer in with a shimmer skeleton.
2. `POST /api/explain {doc_id, node_id: "n7", level: "standard"}`.
3. `rag.explain_node()`:
   1. Returns the cached answer if `(doc_id, n7, standard)` is in SQLite.
   2. Finds the node, its parent and the central topic → query `"TCP. Transport Protocols. Computer Networks"`.
   3. Embeds the query, gets the 20 most similar chunks from Qdrant (only this document), reranks them with the cross-encoder, keeps 4.
   4. Fills `EXPLAIN_USER` with the 4 chunks and calls Ollama with the `LLMExplanation` schema (retry once if invalid).
   5. Adds `sources` from the 4 chunks' metadata, saves to cache, returns.
4. Browser renders the five sections, marks the node explored (✓ + green ring) and updates "N of M explored".
5. Toggling **Explain simpler** repeats steps 2–4 with `level: "simple"` (cached separately).

---

## 6. Design decisions worth explaining to judges

- **Mind map first, not chat:** the graph is the main screen; the LLM fills in content on demand.
- **Structured outputs:** the JSON Schema is sent to Ollama, so the model can only produce valid shapes. Pydantic still double-checks, and there is one retry.
- **Two-stage retrieval:** fast vector search (top 20) then an accurate cross-encoder (top 4). Better context means better explanations.
- **Sources from metadata, not the LLM:** page/slide chips can't be hallucinated.
- **Fallback map:** if the LLM is down, the user still gets a map built from headings and keywords.
- **Cache:** second clicks are instant and the demo is predictable.
- **Background processing + polling:** the upload request returns instantly, and the UI shows real progress.
- **Offline:** models are baked into the Docker image; Ollama runs locally.

---

## 7. Glossary

- **Embedding:** a list of numbers (here 384) representing the *meaning* of a text. Similar meanings give similar vectors, so we can search by meaning instead of exact words.
- **Chunk:** a short piece of a document (~1000 characters). Documents are split into chunks because models handle short passages better and search can pinpoint the exact relevant part.
- **Chunk overlap:** consecutive chunks share 150 characters, so a sentence cut at a boundary still appears whole in one of them.
- **Vector database (Qdrant):** a database built to store embeddings and quickly find the ones closest to a query vector.
- **Cosine similarity:** how closely two vectors point in the same direction (1 = same meaning, 0 = unrelated).
- **RAG (Retrieval-Augmented Generation):** first retrieve relevant passages from your own documents, then let the LLM write an answer using only those passages. This reduces made-up answers.
- **Reranker (cross-encoder):** a model that reads the question and a passage *together* and scores relevance. It is more accurate than comparing embeddings but slower, so it only reranks the top 20.
- **LLM (Large Language Model):** the text-generating AI model (here Llama 3.1 8B) running locally through **Ollama**.
- **Structured output:** asking the LLM to answer in a fixed JSON shape (a JSON Schema). Ollama constrains generation so the output always parses.
- **JSON Schema:** a description of what valid JSON looks like (fields, types, list lengths), generated automatically from our Pydantic models.
- **Pydantic:** Python library that validates data against classes; FastAPI uses it for requests/responses.
- **YAKE:** a statistics-based keyword extractor that needs no training or internet.
- **Context window (`num_ctx`):** how many tokens the LLM can read at once. Prompts are kept small enough to fit.
- **Temperature:** randomness of the LLM. Low (0.2) gives focused, repeatable answers.
- **Background task:** work FastAPI runs after sending the response, so the user doesn't wait.
- **Polling:** the browser asking "are you done yet?" on a timer (every 1.5 s).
