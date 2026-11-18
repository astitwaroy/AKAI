# VectorDB

A vector database I wrote from scratch in C++, with a small web UI so you can actually see what it's doing.

I made this because I wanted to know how vector databases like Pinecone, Weaviate and Chroma really work. Most tutorials import a library and stop there, so I wrote the indexing and search myself. It has three search algorithms you can run against each other (HNSW, KD-Tree and brute force), and a small RAG setup that uses a local LLM through Ollama to answer questions about your own notes.

It's a learning project, not something you'd put in production, but everything in it works.

## What I used

- C++ for the whole backend. The three indexes, the REST API and the RAG logic all live in `main.cpp`.
- cpp-httplib (`httplib.h`), a single-header HTTP library. It runs the server and also handles the calls out to Ollama.
- Ollama, running locally. `nomic-embed-text` turns text into 768-dimensional vectors, and `llama3.2` writes the answers.
- A single `index.html` for the frontend: the search page, the scatter plot, the documents page and the chat.
- g++ from MSYS2 (UCRT64) to compile it on Windows.

## What it can do

Search works on two kinds of data.

The first is a set of 20 pre-loaded demo vectors. Each one has 16 dimensions and belongs to one of four topics: computer science, math, food or sports. They're there so you can try search straight away without setting up Ollama. The UI squashes them down to 2D with PCA and plots them, and you can see each topic sitting in its own cluster. That picture is pretty much what "semantic similarity" means.

For the demo vectors you can pick the algorithm (HNSW, KD-Tree or brute force) and the distance metric (cosine, Euclidean or Manhattan), or run all three algorithms on the same query and compare how fast they are.

The second is real text. Paste in anything, like lecture notes or a Wikipedia article, and Ollama embeds it. Those embeddings go into their own HNSW index, separate from the demo vectors, and that's what the RAG part searches.

## How a question gets answered

When you add a document:

1. Long text is split into chunks of about 250 words. The chunks overlap a little so an idea sitting on the boundary doesn't get cut in half.
2. Each chunk is sent to Ollama, and `nomic-embed-text` gives back a 768-number vector.
3. That vector is added to the HNSW index.

When you ask a question:

1. The question is embedded the same way.
2. HNSW finds the 3 chunks closest to it.
3. Those chunks go to `llama3.2` along with your question, and it answers using only that text.

On the Ollama side it only uses two endpoints, `/api/embeddings` and `/api/generate`.

## Using the UI

### Search tab

Type a concept like `binary tree`, `sushi`, `basketball` or `calculus`, choose an algorithm and a distance metric, and hit Search. You get the nearest vectors with their distances, and the match lights up on the scatter plot. Compare All Algos runs the same query through all three algorithms so you can see the speed difference.

### Documents tab

Give the document a title (something like `Operating Systems Notes`), paste the text and click Embed & Insert. Long documents get chunked automatically. The very first time can be slow because Ollama downloads the embedding model on first use.

### Ask AI tab

Add at least one document first, otherwise there's nothing to search. Then type a question and click Ask AI. The answer types itself out, and the small chips under it show which chunks it used, so you can check where the answer came from.

## REST API

The server runs on `http://localhost:8080`.

For the demo vectors:

- `GET /search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` does a k-nearest-neighbour search
- `POST /insert` adds a demo vector
- `DELETE /delete/:id` removes one by ID
- `GET /items` lists all the demo vectors
- `GET /benchmark?v=...&k=5&metric=cosine` runs a query through all three algorithms and compares them
- `GET /hnsw-info` shows the HNSW graph structure and stats for each layer
- `GET /stats` gives general database stats

For documents and RAG:

- `POST /doc/insert` with `{"title":"...","text":"..."}` embeds and stores a document
- `GET /doc/list` lists the stored documents
- `DELETE /doc/delete/:id` deletes a document chunk
- `POST /doc/ask` with `{"question":"...","k":3}` pulls the context and generates an answer
- `GET /status` shows whether Ollama is up and which models are set

A search from PowerShell:

```powershell
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

Asking a question:

```powershell
curl -X POST http://localhost:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is dynamic programming?","k":3}'
```

If you're on the older Windows PowerShell, `curl` is an alias for `Invoke-WebRequest` there, so type `curl.exe` instead.

## The code

There are only four files: `main.cpp` (the whole backend), `httplib.h` (cpp-httplib), `index.html` (the frontend) and this README.

Inside `main.cpp`, `BruteForce`, `KDTree` and `HNSW` are the three indexes. `VectorDB` puts all three behind one interface for the 16D demo vectors. `DocumentDB` is a separate HNSW-only index for the 768D Ollama embeddings, and `OllamaClient` is the small HTTP client that talks to Ollama.

## Notes on the algorithms

### Brute force

It compares the query against every single vector. That's O(N·d) and always exact. It's the baseline I measure the other two against.

### KD-Tree

It splits space in half again and again, one dimension at a time, cycling through the dimensions as the tree gets deeper. While searching, it skips a whole branch when even the closest point that branch could possibly hold is further away than the best match found so far.

It's exact and close to O(log N) when there aren't many dimensions. The catch is that it falls apart as the dimensions go up. Up to around 20 it's fine, but at 768 it ends up checking nearly everything and is barely better than brute force. In high dimensions almost all of the space sits near the edges, so the distance bounds it relies on to skip branches hardly ever rule anything out.

### HNSW (Hierarchical Navigable Small World)

This is the one that matters. It's the same kind of graph index that Weaviate, Chroma and Milvus use.

Every vector is a node in a graph with several layers. When a node is inserted it's given a random top layer, and the odds drop off fast, so each layer up has far fewer nodes than the one below. Layer 0 has every node, with lots of short links. The upper layers are sparse and their links cover longer distances.

To insert a node, start at the top layer and walk greedily towards the new vector, then drop down a layer and do it again. From the node's own top layer down to layer 0, run a wider search (ef_construction = 200) and link it both ways to its M nearest neighbours.

Searching follows the same path down. Once it gets to layer 0, it widens out and keeps the ef best candidates in a priority queue.

The way I think about it, the upper layers are highways that get you to the right part of town quickly, and layer 0 is the local streets where you find the exact house. That's how it gets to roughly O(log N). It's approximate, so now and then it can miss the true nearest neighbour, but it's so much faster that the trade is worth it. It also doesn't split space along axes the way the KD-Tree does, so high dimensions don't break it.

## If something goes wrong

- The header says `Ollama: OFFLINE`. Ollama isn't running, so start it with `ollama serve`.
- Embedding takes forever the first time. Ollama is downloading the model, give it a couple of minutes.
- `g++: command not found`. Add `C:\msys64\ucrt64\bin` to your Windows PATH.
- Port 8080 is already in use. Find what's using it with `netstat -ano | findstr 8080`, then kill it with `taskkill /PID <pid> /F`.
- Answers are slow. That's normal on a laptop CPU, where `llama3.2` can take 10 to 30 seconds. The smaller model below helps.

### Using a smaller model

If `llama3.2` is too slow on your machine, pull the 1B version:

```powershell
ollama pull llama3.2:1b
```

Then change the line in `main.cpp` where `genModel` is set:

```cpp
std::string genModel = "llama3.2:1b";
```

Recompile and restart the server.

## License

MIT. Do whatever you want with it.
