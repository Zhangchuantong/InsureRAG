# InsureRAG - 保险条款智能问答系统

InsureRAG 是面向保险条款解析与理赔规则问答的本地化 RAG 服务，支持 PDF 质量预检、OCR fallback、父子分块、BGE-M3 Dense/Sparse 混合检索、RRF 融合、BGE-Reranker 重排、vLLM/Qwen3 本地生成、FastAPI、MCP Server、Dashboard、Trace 和 RAGAS 评测。

本项目定位为个人学习与面试展示项目，不宣称生产级准确率。

## 系统架构

```text
PDF / 扫描件
  -> PDF 质量预检
  -> OCR fallback
  -> 父子分块 + metadata enrichment
  -> BGE-M3 Dense/Sparse Embedding
  -> Milvus Standalone
  -> Dense/Sparse Hybrid Retrieval
  -> RRF 融合
  -> BGE-Reranker 精排
  -> Parent Dedup 父块去重
  -> vLLM / Qwen3 本地生成
  -> FastAPI / MCP Server
  -> Dashboard / Trace / RAGAS Evaluation
```

## 技术栈

- Python 3.12
- LangChain / PyPDFLoader
- RapidOCR / ONNXRuntime / PyMuPDF
- BGE-M3 Embedding
- Milvus Standalone
- RRF 排名融合
- BGE-Reranker CrossEncoder
- vLLM + Qwen3-8B-AWQ
- OpenAI-Compatible API
- FastAPI / MCP Server
- RAGAS
- pytest

## 配置说明

主要配置文件：

- [config/config.yaml](config/config.yaml)
- [.env.example](.env.example)

配置优先级：

```text
内置默认值 < config/config.yaml < .env / 环境变量
```

常用配置：

```text
llm.base_url
llm.model
vector_db.host
vector_db.port
vector_db.collection
retrieval.dense_top_k
retrieval.sparse_top_k
retrieval.final_top_k
reranker.enabled
ocr.enabled
trace.output_dir
api.port
dashboard.port
cache.enabled
cache.backend
cache.redis_url
cache.semantic_direct_threshold
cache.semantic_answer_threshold
cache.semantic_retrieval_threshold
```

`.env.example` 不包含真实 API key。本地 vLLM 默认使用 `EMPTY` 占位。

## 环境准备

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

建议使用项目当前环境：

```bat
D:\software\conda\envs\Agent_work\python.exe
```

## 启动 Milvus

```bash
docker compose -f docker-compose.milvus.yml up -d
```

默认连接地址：

```text
http://localhost:19530
```

## 启动 Redis 缓存

```bash
docker compose -f docker-compose.redis.yml up -d
```

默认连接地址：

```text
redis://localhost:6379/0
```

如果 Redis 没有启动，且 `cache.fallback_to_json=true`，系统会自动退回 `data/cache.json`，保证 RAG 服务仍然可以运行。

## 启动 vLLM / Qwen3

项目默认使用：

```text
base_url: http://localhost:8002/v1
model: Qwen/Qwen3-8B-AWQ
```

示例：

```bash
docker run --gpus all --name vllm-qwen3-8b-awq ^
  -p 8002:8000 ^
  -v /path/to/huggingface-cache:/root/.cache/huggingface ^
  vllm/vllm-openai:latest ^
  --model Qwen/Qwen3-8B-AWQ ^
  --trust-remote-code ^
  --gpu-memory-utilization 0.85 ^
  --max-model-len 8192
```

## 文档入库

把保险条款 PDF 放到 `data/` 目录，然后执行：

```bash
python stage1_load_split.py
python stage2_build_db.py
```

> 注意：当前为**重建式入库**——`stage2_build_db.py` 每次运行都会 drop 并重建整个 collection，对全部文档重新索引，**不支持单文档增量插入/更新/删除**。这是为演示简化的取舍；生产级多文档管理需要按 `document_id` upsert 与局部重建。

入库阶段会生成：

- `data/chunks.json`
- `traces/ingestion/*.json`

ingestion trace 会记录：

- document
- parse_method
- ocr_used
- quality_score
- text_density
- page_count
- parent_chunks
- child_chunks
- embedding_status
- milvus_insert_status
- warnings
- error

## 启动 FastAPI

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

也可以构建 API 镜像：

```bash
docker build -t insurerag-api:latest .
docker run --rm -p 8000:8000 --env-file .env insurerag-api:latest
```

或使用示例 compose：

```bash
docker compose -f docker-compose.api.yml up --build
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

### 两种运行方式与 Docker localhost 注意事项

依赖地址（vLLM / Milvus / Redis）在两种运行方式下不一样：

| 运行方式 | vLLM / Milvus / Redis 地址 |
|---|---|
| **本机直接 `python` 运行** | 用 `localhost`（默认配置即可，无需改动）|
| **API 跑在 Docker 容器里、依赖在宿主机** | 用 `host.docker.internal` |
| **API 与依赖在同一个 compose 网络** | 用各自的 service name（如 `redis`、`milvus-standalone`）|

> ⚠️ **容器内的 `localhost` 指向容器自身，不是宿主机。** 如果 API 在容器里却用 `localhost:19530` 连 Milvus，会连到容器自己、必然失败。

容器访问宿主机服务时这样覆盖（环境变量或 `.env`）：

```env
INSURERAG_LLM_BASE_URL=http://host.docker.internal:8002/v1
INSURERAG_VECTOR_DB_HOST=host.docker.internal
INSURERAG_CACHE_REDIS_URL=redis://host.docker.internal:6379/0
```

- **Windows / macOS**：Docker Desktop 自带 `host.docker.internal`，开箱即用。
- **Linux**：需要给容器加 `--add-host=host.docker.internal:host-gateway`（`docker-compose.api.yml` 里已用 `extra_hosts` 配好）。

`docker-compose.api.yml` 是一份只起 API 容器的示例，默认通过 `host.docker.internal` 访问宿主机上的 vLLM/Milvus/Redis；它**不改动** `docker-compose.milvus.yml` / `docker-compose.redis.yml` 的端口。

### Web 界面（上传 + 问答）

主 FastAPI app 内置了一个轻量前端（`web/index.html`），浏览器端通过 `fetch` 调用 `/api/*`，访问：

```text
http://127.0.0.1:8000/
```

流程：上传一份保险条款 PDF → 后台自动解析/分块/向量化并建立**独立检索库**（`upload_<hash>` collection）→ 入库完成后即可针对这份 PDF 问答。每份上传文档进入各自的 collection，互不影响（重新上传同一文件只重建它自己的 collection）。

相关端点：

```text
POST /api/upload                          上传 PDF，后台异步入库，返回 document_id 与 collection
GET  /api/upload/{document_id}/status     轮询入库状态（processing / ready / failed）
POST /api/query  body 带 collection       针对指定 PDF 的 collection 问答
```

入库较慢（CPU 上 BGE-M3 向量化），前端通过状态轮询展示进度。若开启了鉴权，前端需在页面顶部填入 `X-API-Key`。

### API 示例

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

`/health` 会检查 Redis、Milvus、vLLM、Embedding 和 Reranker 状态，并返回每个依赖的 `status`、耗时和错误码。其中 Embedding/Reranker 仅上报模型是否已加载（模型在首次查询时才懒加载），健康检查本身不会触发模型加载，保证探针轻量、可安全用于容器存活探测。

如果开启 API Key 鉴权：

```env
INSURERAG_API_AUTH_ENABLED=true
INSURERAG_API_KEY=your-private-api-key
```

业务接口需要带请求头：

```text
X-API-Key: your-private-api-key
```

`/health` 保持不鉴权，便于容器健康检查。

检索 + 生成：

```bash
curl -X POST http://127.0.0.1:8000/api/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"等待期内生病了能赔吗？\",\"collection\":\"insure_rag\",\"top_k\":5}"
```

只检索：

```bash
curl -X POST http://127.0.0.1:8000/api/search ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"等待期内生病了能赔吗？\",\"collection\":\"insure_rag\",\"top_k\":5}"
```

查看查询 trace：

```text
GET /api/traces/{trace_id}
```

查看 ingestion trace：

```text
GET /api/ingestion-traces
```

查看聚合指标：

```text
GET /api/metrics
```

答案由 vLLM **流式生成**，因此每次查询都会采集生成性能指标：

```text
ttft_ms            首 token 延迟（第一个字吐出来用了多久）
tokens_per_sec     输出吞吐（生成速度）
prompt_tokens      输入 token 数
completion_tokens  输出 token 数
```

单次查询的这些值直接出现在 `POST /api/query` 响应里；`GET /api/metrics` 还会聚合出 `avg_ttft_ms`、`p95_ttft_ms`、`avg_tokens_per_sec`、`avg_completion_tokens`、`avg_prompt_tokens`。缓存命中不经过生成，这些字段为空。

## 缓存机制

项目默认使用 Redis 缓存，并保留本地 JSON fallback：

```text
Redis: redis://localhost:6379/0

JSON fallback: data/cache.json
```

## 可观测性与错误分类

每次 `/api/query` 会写入请求级结构化日志，字段包括：

```text
trace_id, question, latency_ms, cache_hit, cache_type, retrieval_mode, rerank_mode, generation_mode, error_type
```

日志位置：

```text
logs/insurerag.log
```

查询 trace 默认持久化到 SQLite（`traces/traces.db`）；`GET /api/traces/{trace_id}`、`GET /api/metrics` 与 Dashboard 指标均由该库做索引查询 / 单次聚合，不再逐个 JSON 文件读盘。按配置自动清理：

```yaml
trace:
  output_dir: traces
  retention_days: 7   # 超过保留天数的 trace 会被删除
  max_files: 1000     # 最多保留的 trace 条数（超出按时间裁剪）
```

ingestion trace（文档入库记录）仍以 JSON 形式保存在 `traces/ingestion/`。

统一错误码包括：

```text
PDF_PARSE_ERROR
OCR_ERROR
MILVUS_SEARCH_ERROR
RERANKER_ERROR
LLM_TIMEOUT
CACHE_ERROR
```

Redis is the preferred cache backend. The cache keeps exact answer cache,
semantic answer cache, and semantic retrieval cache. If Redis is down and
`cache.fallback_to_json=true`, the project automatically falls back to the
local JSON cache so demos do not fail because of cache infrastructure.

缓存分三层：

1. Exact Answer Cache：规范化后一模一样的问题直接返回缓存答案。
2. Semantic Answer Cache：相似度达到 `cache.semantic_direct_threshold`，且 collection、模型、top_k、检索配置一致，直接返回缓存答案。
3. Semantic Retrieval Cache：相似度达到 `cache.semantic_retrieval_threshold` 但未达到直接答案阈值时，复用缓存证据并重新调用 LLM 生成。

当相似度达到 `cache.semantic_answer_threshold` 时，系统会重新检索当前问题，并检查：

- 证据 Jaccard overlap 是否达到 `cache.evidence_overlap_threshold`
- top1 evidence 是否一致
- 缓存答案是否为正常 LLM 生成
- 缓存答案是否无 errors / fallback

如果缓存中没有相似度大于 `cache.semantic_retrieval_threshold` 的问题，本次正常 RAG 完成后会自动写入语义缓存。

API 返回会包含：

```json
{
  "cache_hit": true,
  "cache_type": "exact_answer",
  "similarity": null,
  "evidence_overlap": null
}
```

## 启动 MCP Server

MCP Server 使用 stdio 方式，适合被外部 Agent 调用：

```bash
python -m mcp_server.server
```

MCP tools：

- `query_insurance_clause`
- `search_related_clauses`
- `get_clause_detail`
- `list_documents`

示例 MCP client 配置：

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

更多说明见 [MCP.md](MCP.md)。

## 启动 Dashboard

Dashboard 用于展示 Overview、Metrics、Document Browser 和 Query Trace：

```bash
python -m dashboard.app
```

默认地址：

```text
http://127.0.0.1:8001
```

## 运行测试

安装测试依赖：

```bash
pip install pytest
```

运行 unit tests：

```bash
pytest tests/unit -q
```

运行全部测试：

```bash
pytest -q
```

测试包含：

- PDF 质量预检
- OCR fallback 触发条件
- 父子分块 metadata
- RRF 排名融合
- Qwen thinking 过滤
- 检索 fallback
- FastAPI `/api/query` 响应结构
- 缓存底层逻辑（归一化 / 余弦相似度 / 证据 Jaccard overlap / TTL / 语义匹配护栏）
- 缓存三层路由（精确 / 语义直答 / 证据校验 / 语义检索复用 / miss）

## RAGAS 评测

评测集在 [eval/eval_set.json](eval/eval_set.json)，按文档分组、每组指定 `collection`。
其中 CLI 入库的康宁用固定的 `insure_rag`；平安那份是网页上传的，`collection` 是文件名哈希
（`upload_<hash>`）。**若重新上传平安或换了机器，需要把该组的 `collection` 改成实际值**；
`stage6` 在跑前会校验 collection 是否存在，缺失时会醒目警告并跳过该组（不再静默 0 召回）。

> 注：`llm.enable_thinking` 默认**关闭**（评测显示对抽取式条款问答质量基本无影响，但延迟约减半）。

运行：

```bash
python stage6_evaluate.py
```

评测指标包括：

- Answer Correctness
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

当前评测集规模有限，指标只能用于本地样本对比，不应夸大为生产准确率。

## 项目限制

- Milvus 使用本地 Standalone，不是生产集群。
- 评测集规模有限，不能代表真实线上效果。
- 仅提供单一 API Key 鉴权（`X-API-Key`），未包含多用户权限体系（RBAC）。
- 尚未包含生产级监控、限流和并发压测。
- Dashboard 是轻量本地展示，不是完整前端系统。
- 多 collection / 多文档管理是基础能力，暂不包含复杂权限和生命周期管理。
