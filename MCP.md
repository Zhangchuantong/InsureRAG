# InsureRAG MCP Server

InsureRAG now exposes its insurance clause retrieval and RAG question answering abilities as an MCP Server. The server uses stdio transport first, so it can be called by external Agents and MCP clients without starting FastAPI.

FastAPI and MCP are not coupled. Both adapters call the shared service layer in `services/rag_service.py`.

## Start

Run from the project root:

```bash
D:\software\conda\envs\Agent_work\python.exe -m mcp_server.server
```

If your shell has already activated the `Agent_work` conda environment:

```bash
python -m mcp_server.server
```

## Tools

### query_insurance_clause

Answer a natural language insurance question and return evidence plus `trace_id`.

Input:

```json
{
  "question": "等待期内生病了能赔吗？",
  "collection": "insure_rag"
}
```

Output:

```json
{
  "ok": true,
  "answer": "...",
  "evidence": [
    {
      "content": "...",
      "source": "保险条款.pdf",
      "page": null,
      "clause": null,
      "score": 0.87,
      "parent_id": 3,
      "id": 12
    }
  ],
  "trace_id": "...",
  "latency_ms": 5230,
  "collection": "insure_rag"
}
```

### search_related_clauses

Only search related clauses. It does not call the LLM.

Input:

```json
{
  "query": "等待期内生病了能赔吗？",
  "top_k": 5
}
```

### get_clause_detail

Return a full parent clause from `data/chunks.json` by `clause_id`, numeric parent id, or article label such as `第三条`.

Input:

```json
{
  "clause_id": "第三条"
}
```

### list_documents

Return current knowledge base documents and chunk count.

Input:

```json
{}
```

Output:

```json
{
  "ok": true,
  "documents": ["保险条款.pdf"],
  "chunk_count": 54,
  "collection": "insure_rag"
}
```

## Client Configuration Example

The exact config format depends on the MCP client. A common stdio-style config looks like this:

```json
{
  "mcpServers": {
    "insurerag": {
      "command": "D:\\software\\conda\\envs\\Agent_work\\python.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "D:\\code\\PythonProject\\InsureRAG"
    }
  }
}
```

Before using `query_insurance_clause` or `search_related_clauses`, make sure Milvus Standalone and vLLM/Qwen3 are running. `list_documents` and `get_clause_detail` only depend on local files.
