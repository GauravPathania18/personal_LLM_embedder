# Personal LLM Embedding Service
Text-to-vector embedding API for LLMs built with FastAPI and SentenceTransformers.

## Features
- Convert single or multiple texts into vector embeddings
- Automatic HTML stripping and whitespace normalization
- CPU/GPU support via DEVICE environment variable
- Easy to extend for OpenAI/Gemini embeddings

## Setup

1. Clone the repo:

```bash
git clone https://github.com/<your-username>/personal-llm-embedder.git
cd personal-llm-embedder
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the service:

```bash
uvicorn embedder_api:app --reload --host 0.0.0.0 --port 8000
```

4. Check the health endpoint:

```
http://localhost:8000
```
API Endpoints:

GET / :
   Returns service status:
   ```json
   {
  "status": "ok",
  "mode": "local",
  "model": "all-mpnet-base-v2",
  "device": "cpu",
  "html_strip": true
   }
   ```
POST /embed :
   Generate embeddings:
   Single text example:
   ```json
  {
  "text": "Hello world",
  "source": "user_prompt"
  }
  ```
   Multiple texts example:
   ```json
   {
  "texts": ["Hello", "world"],
  "source": "user_prompt"
   }
   ```
Response example:
```json
{
  "mode": "local",
  "model_name": "all-mpnet-base-v2",
  "vector_size": 768,
  "count": 2,
  "processing_time_sec": 0.123,
  "items": [
    {
      "id": "uuid",
      "vector": [...],
      "metadata": {
        "text": "Hello",
        "timestamp": "2025-08-15T00:00:00",
        "source": "user_prompt",
        "embedding_mode": "local"
      }
    }
  ]
}
```

5. Environment Variables:
```
export EMBEDDING_MODE=local
export LOCAL_MODEL=all-mpnet-base-v2
export DEVICE=cpu
export HTML_STRIP=true
```



