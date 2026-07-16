from typing import List

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.bge_client_util import get_bgem3_client, generate_vector_data
from knowledge.utils.milvus_client_util import get_milvus_client, create_vector_search_request, \
    execute_bybrid_search_query

# 向量检索的节点
class VectorSearchNode(BaseNode):
    def process(self,state:QueryGraphState)->QueryGraphState:
        # 1 从上一步节点state获取重写问题 和 item_name
        item_names,rewritten_query = self.validate_data(state)
        if not item_names:
            # item_names 为空，向量搜索无意义，返回空结果
            return {"embedding_chunks": []}


        # 2 对rewritten_query重写问题生成两个向量
        # 获取bgem3模型对象
        bgem3_client = get_bgem3_client()
        # 获取milvus对象
        milvus_client = get_milvus_client()
        # 调用方法生成两个向量
        embedding_result = (
            generate_vector_data(bgem3_client,[rewritten_query]))
        if not embedding_result:
            return {"embedding_chunks": []}

        # 3 对item_names标量字段构建过滤条件
        # item_name in ["商品名1" , "商品名2"]
        itemname_filter_expr =self.create_itemname_filter(item_names)

        # 4 构建rewritten_query向量条件
        vector_search_request = create_vector_search_request(
            dense_vector= embedding_result['dense'][0],
            sparse_vector= embedding_result['sparse'][0],
            expr=itemname_filter_expr
        )

        # 5 调用工具类方法，执行查询
        res = execute_bybrid_search_query(
            milvus_client=milvus_client,
            collection_name="kb_chunks_v2",
            search_request=vector_search_request,
            norm_score=True,
            output_fields=["chunk_id","content","item_name"]
        )

        # 6 返回结果，处理，更新到state，返回state
        if not res or not res[0]:
            return {"embedding_chunks": []}
        # 更新到state
        # state["embedding_chunks"] = res[0]
        # return state
        return {"embedding_chunks":res[0]}


    # 3 对item_names标量字段构建过滤条件
    def create_itemname_filter(self,item_names:List[str])->str:
        item_name_filter = ", ".join(f'"{name}"' for name in item_names)
        return f" item_name in [{item_name_filter}]"

    # 1 从上一步节点state获取重写问题 和 item_name
    def validate_data(self,state:QueryGraphState):
        item_names = state.get("item_names")
        if not item_names:
            # item_names 为空时返回空结果，不阻断流程（联网搜索可以处理）
            return [], state.get("rewritten_query")

        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise Exception(f"Rewritten query field is empty")

        return item_names,rewritten_query

