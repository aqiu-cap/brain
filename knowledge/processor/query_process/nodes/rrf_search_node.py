import json
from typing import List, Tuple, Dict, Any

from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.state import QueryGraphState


# RRF融合节点
class RRFSearchNode(BaseNode):
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 获取向量查询结果 和 hyde查询结果
        embedding_chunks = state.get("embedding_chunks")
        hyde_embedding_chunks = state.get("hyde_embedding_chunks")

        # 构建字典类型数据
        # {
        #  "vector_search_result": ([{向量chunk1},{向量chun2}] , 1.0),
        #  "hyde_search_result":  ([{hyde的chunk1},{hyde的chun2}] , 1.0)
        # }
        search_source = {
            "vector_search_result":
                  (self.get_chunklist(embedding_chunks) , 1.0),
            "hyde_search_result":
                  (self.get_chunklist(hyde_embedding_chunks) , 1.0),
        }
        # search_source处理
        # [
        #   ([{向量chunk1},{向量chun2}] , 1.0),
        #   ([{hyde的chunk1},{hyde的chun2}] , 1.0)
        # ]
        rrf_inputs = list(search_source.values())

        # 调用方法
        # 把rrf_inputs列表里面每个文档，使用rrf公式计算分数
        # 按照分数从高到低排序，返回排序之后结果
        """
            [
                (
                    {
                        "content":111
                        "chunk_id": 1
                    },
                    
                    0.0162
                )
            ]
        """
        rrf_merge_result:List[Tuple[Dict[str,Any],float]] = (
                                  self.rrf_merge(rrf_inputs))

        # 更新到state里面
        # [{内容1},{内容2}]
        rrf_chunks = [chunk for chunk,score in rrf_merge_result]
        state["rrf_chunks"] = rrf_chunks
        return state

    # [{向量chunk1},{向量chun2}]
    def get_chunklist(self,inputs)->List[Dict[str,Any]]:
        result = []
        for input in inputs:
            entity = input.get("entity")
            if not entity:
                continue
            result.append(entity)
        return result

    # 把rrf_inputs列表每个文档获取到
    # 计算每个文档分数，（如果一个文档在多种查询都出现了，分数叠加）
    # 根据分数，按照从高到低排序
    ### 传递数据格式：
    # [
    #   ([{向量chunk1},{向量chun2}] , 1.0),
    #   ([{hyde的chunk1},{hyde的chun2}] , 1.0)
    # ]
    ### 返回数据格式
    # [ ({content:11,item_name:22,chunk_id:1} , 0.0162)]
    def rrf_merge(self,rrf_inputs):
        # 定义字典
        # {"chunk_id":分数}
        chunk_score = {}
        # 定义字典
        # {"chunk_id":文档内容 }
        chunk_doc = {}

        # 遍历 rrf_inputs列表
        # ( [{向量chunk1},{向量chun2}] , 1.0)
        for rrf_input,weight in rrf_inputs:
            ##  rrf_input    [{向量chunk1},{向量chun2}]
            ## weight    1.0
            # 把遍历元组第一个列表继续遍历 rrf_input
            # [{向量chunk1},{向量chun2}]
            for index,doc in enumerate(rrf_input):
                # index就是文档排名，从0开始，排名从1开始
                rank = index+1

                # 获取chunk_id
                # 获取chunk_id目的：为了后面如果相同文档在不同检索里面出现，叠加
                chunk_id = doc.get("chunk_id")
                if not chunk_id:
                    continue

                # rrf公式计算文档分数
                #  权重 / 60 + 文档排名
                # doc_score = weight / (60 + rank)
                # 判断当前文档之前是否计算过分数，如果计算过获取之前分数 和 当前分数叠加
                chunk_score[chunk_id] = (
                    chunk_score.get(chunk_id,float(0)) + weight / (60 + rank)
                )

                # 放chunk_id 和 内容
                # setdefault: 相同key数据只能放一次
                chunk_doc.setdefault(chunk_id,doc)

        # 根据分数进行排序，返回 内容+分数
                # {"chunk_id":分数}
                # chunk_score = {}

                # # 定义字典
                # # {"chunk_id":文档内容 }
                # chunk_doc = {}
        # 返回 [(文档内容1,分数),(文档内容1,分数)]  10条记录
        sorted_result = sorted(
            [(chunk_doc[chunk_id], score)
              for chunk_id,score in chunk_score.items()
            ],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_result[:10]

if __name__ == "__main__":
    # 模拟两路检索结果
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    rrf_node = RRFSearchNode()
    result = rrf_node.process(mock_state)
    print(json.dumps(result,indent=4,ensure_ascii=False))
