# 智能知识库问答系统
基于 RAG的知识库问答系统。支持上传PDF、MD文档构建知识库，通过向量搜索+联网搜索回答问题。

## 1.环境准备
### 技术栈
Python 3.10+
框架 ：FastAPI + LangGraph
向量数据库 ：Milvus 2.4+
文档数据库 ：MongoDB
对象存储 ；MinIO
嵌入模型 ：BGE-M3
重排序模型 ：BGE Reranker
LLM ：阿里云 DashScope（通义千问）
联网搜索 ：阿里云 MCP WebSearch

### 安装依赖：
cd knowledge
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

### 启动服务
导入服务：python knowledge/front/api/import_router.py
查询服务：python knowledge/front/api/query_router.py
API 接口
上传页面 ：/import
聊天页面 ：/chat
## 2.运行截图
导入文档
<img width="1199" height="1374" alt="image" src="https://github.com/user-attachments/assets/7c97fd7b-e7b6-4856-8ffd-945c0891c053" />
用户查询
<img width="2541" height="1403" alt="image" src="https://github.com/user-attachments/assets/7cfcab32-d1bf-4766-b4da-2f1510b3221f" />
