# DataGouv Alive — Backend MVP

A lightweight, production-ready FastAPI backend for the "DataGouv Alive" hackathon project. It uses Gemini-based agents to orchestrate queries, retrieve datasets from data.gouv.fr (via an MCP layer), and synthesize grounded answers.

## Architecture Summary

```
backend/
├── app/
│   ├── main.py            # FastAPI app factory + routers
│   ├── config.py          # Centralised settings (pydantic-settings)
│   ├── schemas.py         # Pydantic request/response models
│   ├── prompts/           # Agent prompt templates (loaded from disk)
│   │   ├── orchestrator.txt
│   │   ├── dataset_scout.txt
│   │   └── answer_synthesizer.txt
│   ├── routes/
│   │   ├── health.py      # GET /health
│   │   ├── demo.py        # GET /demo/scenarios
│   │   └── query.py       # POST /query
│   ├── services/
│   │   ├── gemini_service.py   # Google GenAI SDK wrapper
│   │   ├── mcp_service.py      # data.gouv.fr MCP abstraction
│   │   └── orchestrator.py     # 3-agent pipeline coordinator
│   └── utils/
│       └── logger.py
├── requirements.txt
├── .env.example
└── README.md
```

**Agents:**
- **Orchestrator** – understands the query and decides what to search for.
- **Dataset Scout** – finds relevant data.gouv.fr resources via MCP.
- **Answer Synthesizer** – writes the final answer grounded in selected sources.

## Setup Instructions

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your `GEMINI_API_KEY`.  
   Set `USE_MOCK_GEMINI=false` to use the real Gemini AI.  
   Set `USE_MOCK_MCP=false` once your MCP server is running.

## How to Run Locally

```bash
uvicorn app.main:app --reload --port 8000
```

The interactive API docs are available at <http://localhost:8000/docs>.

## Environment Variables

| Variable          | Default                        | Description                          |
|-------------------|--------------------------------|--------------------------------------|
| `GEMINI_API_KEY`  | *(empty)*                      | Your Google Gemini API key           |
| `USE_MOCK_GEMINI` | `true`                         | Use mocked Gemini responses          |
| `USE_MOCK_MCP`    | `true`                         | Use mocked MCP / data.gouv responses |
| `MCP_SERVER_URL`  | `http://localhost:8000/mcp`    | URL of your MCP server               |

## Example Requests

**Health check:**
```bash
curl http://localhost:8000/health
```

**Demo scenarios:**
```bash
curl http://localhost:8000/demo/scenarios
```

**Main query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest datasets about electric vehicle charging stations in France?"}'
```

## Demo Mode

With the default settings (`USE_MOCK_GEMINI=true`, `USE_MOCK_MCP=true`) the API runs entirely without external services. All three agents return deterministic mock data, making the backend perfect for a live hackathon demo without needing any API keys.

## Where to Plug in Real MCP Integration

Open `app/services/mcp_service.py` and look for the `TODO` comment inside `search_data_gouv`. Replace the commented-out `httpx` block with a real call to your MCP server endpoint.

## Next Steps

1. **Wire real MCP** – implement the `httpx` block in `mcp_service.py`.
2. **Enable real Gemini** – set `USE_MOCK_GEMINI=false` and add your API key.
3. **Restrict CORS** – replace `allow_origins=["*"]` in `main.py` with the frontend URL before going to production.
