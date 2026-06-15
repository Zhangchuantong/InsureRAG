# InsureRAG - 保险条款智能问答系统

InsureRAG 是一个面向保险条款解释与理赔咨询场景的本地 RAG 项目，覆盖 PDF 解析、父子分块、Dense/Sparse 混合检索、RRF 排名融合、CrossEncoder 重排序、本地大模型生成及 RAGAS 自动评测。

## 系统流程

```text
保险条款 PDF
    |
    v
PDF 解析与文本清洗
    |
    v
按“第 X 条”构建父块 + 约 200 字子块
    |
    v
BGE-M3 Dense / Sparse 向量化
    |
    v
Milvus Lite 双路召回 + RRF 融合
    |
    v
BGE-Reranker 精排
    |
    v
Qwen3-8B-AWQ 基于完整父条款生成答案
```

项目还包含意图识别与 Query Rewrite 实验模块。该模块位于 `stage5_intent.py`，当前基础问答与 RAGAS 主评测链路默认使用原始问题检索。

## 核心实现

### PDF 解析与父子分块

- 使用 LangChain `PyPDFLoader` 解析保险条款 PDF。
- 清理部分旧版 PDF 文本层中重复出现的标题和短语。
- 按照“第 X 条”结构划分父块，保留完整条款语义。
- 使用 `RecursiveCharacterTextSplitter` 生成约 200 字、重叠 30 字的检索子块。
- 使用子块进行精确召回，命中后将完整父条款交给生成模型。

对应文件：`stage1_load_split.py`

### 混合检索与重排序

- 使用 BGE-M3 同时生成 Dense 和 Sparse 向量。
- 使用 Milvus Lite 保存文本、来源、父级条款及两类向量。
- 分别执行 Dense/Sparse 检索，并通过 RRF 融合两路排名。
- 召回 Top 20 候选后，使用 `BAAI/bge-reranker-base` CrossEncoder 精排。

对应文件：`stage2_build_db.py`、`stage3_search.py`

> Sparse 检索使用 BGE-M3 生成的稀疏向量，不是独立 BM25。

### 本地答案生成

- 使用 Docker 和 vLLM 部署 `Qwen/Qwen3-8B-AWQ`。
- 通过 OpenAI-Compatible API 接入本地模型。
- 关闭 Qwen3 thinking 输出，避免思考内容进入最终答案。
- 使用结构化 Prompt 约束模型仅依据召回条款回答，证据不足时拒答并标注依据来源。

对应文件：`stage4_generate.py`

### RAGAS 自动评测

- 基于真实保险条款整理 22 条参考问答样本。
- 使用本地 Qwen3-8B 作为 RAGAS 裁判，BGE-M3 作为评测 Embedding。
- 评估 Answer Correctness、Faithfulness、Answer Relevancy、Context Precision 和 Context Recall。
- `RunConfig(max_workers=1)` 限制单卡并发，单项裁判解析失败时记录为 `NaN`，避免整轮评测中断。

最近一次本地评测结果：

| 指标 | 结果 | 有效样本 |
| --- | ---: | ---: |
| Answer Correctness | 0.7408 | 22/22 |
| Faithfulness | 0.8596 | 19/22 |
| Answer Relevancy | 0.8445 | 22/22 |
| Context Precision | 0.9030 | 22/22 |
| Context Recall | 1.0000 | 19/22 |
| 平均端到端延迟 | 4.95 秒 | 22/22 |

上述结果采集时裁判模型上下文为 4096 Token，部分 Faithfulness 和 Context Recall 样本因此未得到有效分数。当前部署已调整为 8192 Token，重新评测后应以最新结果为准。

对应文件：`stage6_evaluate.py`

## 技术栈

- Python、LangChain、PyPDF
- PyMilvus、Milvus Lite
- BGE-M3、BAAI/bge-reranker-base
- Dense/Sparse 混合检索、RRF
- Qwen3-8B-AWQ、vLLM、Docker
- OpenAI-Compatible API、Prompt Engineering
- RAGAS、Hugging Face Embeddings

## 环境准备

建议使用 Python 3.11 或 3.12。

```bash
pip install -r requirements.txt
```

### 启动 Qwen3

项目默认使用以下接口：

```text
BASE_URL=http://localhost:8002/v1
MODEL=Qwen/Qwen3-8B-AWQ
```

参考 Docker 命令：

```bash
docker run --gpus all --name vllm-qwen3-8b-awq \
  -p 8002:8000 \
  -v /path/to/huggingface-cache:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B-AWQ \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

Windows PowerShell 中需要根据本机环境调整换行符和 Hugging Face 缓存路径。

## 运行方式

1. 将有权使用的保险条款 PDF 放入 `data/`。
2. 依次执行：

```bash
python stage1_load_split.py
python stage2_build_db.py
python stage3_search.py
python stage4_generate.py
python stage6_evaluate.py
```

| 脚本 | 功能 |
| --- | --- |
| `stage1_load_split.py` | PDF 解析、清洗与父子分块 |
| `stage2_build_db.py` | 生成 Dense/Sparse 向量并写入 Milvus Lite |
| `stage3_search.py` | 混合检索、RRF 融合与重排序 |
| `stage4_generate.py` | 检索并生成最终答案 |
| `stage5_intent.py` | 意图识别与 Query Rewrite 实验入口 |
| `stage6_evaluate.py` | 本地 RAGAS 自动评测 |

如果 `data/` 中没有 PDF，`stage1_load_split.py` 会使用内置示例条款帮助验证基础流程。

## 数据说明

保险条款 PDF、生成的 `chunks.json`、Milvus Lite 数据库和本地评测结果均被 `.gitignore` 排除，不会上传到公开仓库。

请自行准备具有合法使用权限的 PDF，并重新运行 `stage1_load_split.py` 和 `stage2_build_db.py` 构建本地数据。

## 当前限制

- 当前为命令行原型，尚未提供 Web API、前端、会话管理和流式输出。
- 尚未对父条款重复召回进行去重，较长条款可能重复占用 Top K。
- Query Rewrite 尚未接入主问答和评测链路。
- 评测集规模较小，指标仅用于项目迭代，不代表生产环境效果。
