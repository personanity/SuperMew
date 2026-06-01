# SuperMew-Med: 面向医疗领域的增强 RAG 解决方案

`SuperMew-Med` 是一个基于通用 RAG 框架 [`agent-core`](https://github.com/icey1287/SuperMew/tree/agent-core) 分支，并针对医疗垂直领域进行深度优化的智能知识库项目。它继承了 `agent-core` 的核心能力，并通过引入医疗知识图谱，显著提升了在复杂医疗问答场景下的准确性和可靠性。

## `SuperMew-Med` vs. `agent-core`: 核心区别

虽然 `SuperMew-Med` 与 `agent-core` 共享基础架构，但其核心创新在于**引入了知识图谱作为第二重检索和推理引擎**，专门用于处理医疗领域的精确知识查询。

| 特性 | `agent-core` (通用框架) | `SuperMew-Med` (医疗特化) |
| :--- | :--- | :--- |
| **核心检索方式** | 向量检索 (稠密 + BM25 稀疏) | **向量检索 + 知识图谱混合检索** |
| **知识来源** | 非结构化文档 | 非结构化文档 + **结构化的医疗知识图谱 (Neo4j)** |
| **典型用例** | 通用知识问答 | 药物相互作用查询、基因-疾病关联分析、临床指南解读等 |
| **关键实现** | [`rag_pipeline.py`](backend/rag_pipeline.py) | [`medical_graph_rag_retriever.py`](backend/medical_graph_rag_retriever.py) |

简而言之，`agent-core` 提供了一个强大的、可扩展的 RAG 底座，而 `SuperMew-Med` 则是在这个底座上构建的“专业应用”，通过“向量+图”的双引擎模式，解决了单纯依赖向量检索难以处理的精确、结构化知识查询问题。

## 架构亮点：向量与图的协同工作流

`SuperMew-Med` 的 RAG 流水线 (由 `LangGraph` 驱动) 实现了一套精密的“向量+图”协同工作流：

1.  **实体提取**: 从用户问题中识别出医疗实体（如药物、基因、疾病等）。
2.  **并行检索**:
    *   **向量检索 (广度)**: 在 Milvus 中对海量医疗文献、文档进行快速的语义相似度搜索。
    *   **图谱检索 (深度)**: 根据提取的实体，在 Neo4j 医疗知识图谱中进行精确的关系遍历和事实查找。
3.  **智能上下文合并**: 将向量检索返回的“相关文本片段”与图谱检索返回的“精确事实”进行融合，构建出既有广度又有深度的上下文。
4.  **LLM 生成**: 将融合后的高质量上下文提供给大语言模型，生成准确、可靠且有据可循的答案。

这种架构使得 `SuperMew-Med` 在处理“药物 A 和药物 B 能否同时使用？”或“与特定基因相关的靶向药物有哪些？”这类问题时，表现远超通用 RAG 系统。

## 可量化的 RAG 效果评估

`SuperMew-Med` 同样内置了一套基于 `Ragas` 的评估体系，用于持续监控和优化 RAG 性能。

- **评估脚本**:
    - [`run_ragas_faithfulness_heatmap.py`](run_ragas_faithfulness_heatmap.py): 生成忠实度热力图，量化分析不同模型配置下的幻觉问题。
    - [`run_ragas_performance_vs_retrieval_depth.py`](run_ragas_performance_vs_retrieval_depth.py): 自动化测试检索深度 (`k`) 对 RAG 各项指标的影响。
- **核心评估维度**: Faithfulness (忠实度), Answer Relevancy (答案相关性), Context Precision/Recall (上下文精确率/召回率)。

## 本地部署

### 1. 环境准备
- Python `3.12+`
- 包管理工具: `uv` (推荐) 或 `pip`
- Docker / Docker Compose (用于运行 Milvus 和 Neo4j)

### 2. 安装依赖
```bash
# 使用 uv (推荐)
uv sync
```

### 3. 配置环境
复制 `.env.example` 文件为 `.env`，并填入必要的 API Keys 和数据库连接信息。
```env
# ===== LLM and Embedding =====
ARK_API_KEY=your_ark_api_key
MODEL=your_model_name
BASE_URL=https://your-llm-endpoint/v1
EMBEDDING_MODEL=BAAI/bge-m3

# ===== Milvus Vector Store =====
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

# ===== Neo4j Knowledge Graph =====
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### 4. 启动依赖服务
```bash
# 启动 Milvus 和 Neo4j
docker compose up -d
```
*注意: 首次启动时，您需要运行 [`backend/scripts/build_neo4j_graph.py`](backend/scripts/build_neo4j_graph.py) 脚本来构建和填充医疗知识图谱。*

### 5. 启动应用
```bash
# 启动后端服务
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
- **前端界面**: `http://127.0.0.1:8000/`
- **API 文档**: `http://127.0.0.1:8000/docs`
