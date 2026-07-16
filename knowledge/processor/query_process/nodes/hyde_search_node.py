from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.bge_client_util import generate_vector_data, get_bgem3_client
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import create_vector_search_request, execute_bybrid_search_query, \
    get_milvus_client


# hyde假设文档生成节点
class HydeSearchNode(BaseNode):
    def process(self,state:QueryGraphState)->QueryGraphState:
        # 1 state获取数据
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise Exception(f"HydeSearchNode: No rewritten query found")
        item_names = state.get("item_names")
        if not item_names:
            # item_names 为空时返回空结果，不阻断流程（联网搜索可以处理）
            return {"hyde_embedding_chunks": []}


        # 2 根据用户问题 + item_name构建提示词，调用LLM得到假设性答案
        hyde_document = (
            self.call_llm_generate_answer(rewritten_query,item_names))
        print("=="*50)
        print(hyde_document)
        print("==" * 50)

        # 3 根据用户问题 + llm返回假设性答案拼接字符串
        embedding_document = f"{rewritten_query}\n{hyde_document}"

        # 4 把用户问题 + llm返回假设性答案拼接字符串 生成两个向量
        bgem3_client = get_bgem3_client()

        vector_result = (
            generate_vector_data(bgem3_client, [embedding_document]))

        # item_name in ["商品名1","商品名2"]
        item_name_filter = self.create_item_name_expr(item_names)

        # 5 根据生成向量，构建向量条件 + 标量条件
        vector_search_request = create_vector_search_request(
            dense_vector=vector_result["dense"][0],
            sparse_vector=vector_result["sparse"][0],
            expr=item_name_filter
        )

        # 6 执行混合查询
        milvus_client = get_milvus_client()
        res = execute_bybrid_search_query(
            milvus_client=milvus_client,
            collection_name="kb_chunks_v2",
            search_request=vector_search_request,
            norm_score=True,
            output_fields=["chunk_id","content","item_name"]
        )
        if not res:
            return {"hyde_embedding_chunks": []}
        return {"hyde_embedding_chunks":res[0]}

    # 2 根据用户问题 + item_name构建提示词，调用LLM得到假设性答案
    def call_llm_generate_answer(self,
                                 rewritten_query:str,
                                 item_names:List[str])->str:
        # 构建提示词
        system_prompt = (f"您是一位{item_names}的技术文档领域的专家，"
                         f"主要擅长编写技术文档、操作手册、文档规格说明")
        user_prompt = USER_HYDE_PROMPT_TEMPLATE.format(
            item_hint = item_names,
            rewritten_query= rewritten_query
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        # 调用llm
        llm_client = get_llm_client()
        result = llm_client.invoke(messages)
        # result.content
        response_content = getattr(result,"content","")
        return response_content

    # 构建标量条件的方法
    # # item_name in ["商品名1","商品名2"]
    def create_item_name_expr(self,item_names:List[str])->str:
        # "商品名1", "商品名2"
        item_names_expr = ", ".join(f'"{name}"' for name in item_names)
        # item_name in ["商品名1","商品名2"]
        return f" item_name in [{item_names_expr}]"

