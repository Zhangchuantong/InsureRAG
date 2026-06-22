# InsureRAG FastAPI 最小服务

项目已新增最小可运行 HTTP API，不重写现有 RAG 主链路，不做复杂异步队列、前端、权限认证或多 collection 动态切换。API 层只封装现有 `stage3_search.search()` 和 `stage4_generate.answer_with_trace()`。

## 启动前确认

1. Milvus Standalone 已启动，默认地址为 `http://localhost:19530`。
2. vLLM / Qwen3 已启动，默认地址为 `http://localhost:8002/v1`。
3. 已执行过 `stage1_load_split.py` 和 `stage2_build_db.py`，Milvus 中已有 `insure_rag` collection。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

也可以直接运行：

```bash
python api/main.py
```

## 接口

| Method | Path | 说明 |
| --- | --- | --- |
| `POST` | `/api/query` | 检索 + 生成答案，返回 `answer`、`evidence`、`trace_id`、`latency_ms` |
| `POST` | `/api/search` | 只执行检索，不调用大模型生成 |
| `GET` | `/api/documents` | 返回当前导入文档和 chunk 数量 |
| `GET` | `/api/traces/{trace_id}` | 查看一次查询的完整 trace |
| `GET` | `/health` | 服务健康检查 |

## 示例

```bash
curl -X POST http://localhost:8000/api/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"等待期内生病了能赔吗？\",\"collection\":\"insure_rag\",\"top_k\":5}"
```

当前版本为了保持简单，`collection` 只允许使用配置文件中的 `vector_db.collection`，默认是 `insure_rag`。如果传入其他 collection，会返回 400。
