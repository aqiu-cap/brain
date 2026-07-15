import json
from typing import Dict, List, Any

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.bge_rerank_util import get_reranker_model


# reranker重排序节点
class RerankerSearchNode(BaseNode):
    def process(self,state:QueryGraphState)->QueryGraphState:
        # 获取用户问题
        user_query = state.get("rewritten_query")

        # 获取rrf融合列表 + web搜索列表 合并到一起
        merged_multi_data:List[Dict[str,Any]] = self.merge_multi_data(state)

        # 把用户问题 + 答案 传递给reranker模型，计算每个答案匹配度分数
        # 返回列表，包含答案 + 匹配度分数 ，根据分数排序
        reranker_doc = self.reranker_data_score(user_query, merged_multi_data)

        # 断崖检测
        final_data = self.cliff_data(reranker_doc)

        # 更新
        state["reranked_docs"] = final_data
        return state

    # 获取rrf融合列表 + web搜索列表 合并到一起
    def merge_multi_data(self,state:QueryGraphState)->List[Dict[str,Any]]:
        final_data = []
        # 获取rrf融合列表数据，遍历结果封装
        rrf_chunks = state.get("rrf_chunks")
        for rrf_chunk in rrf_chunks:
            chunk_id = rrf_chunk.get("chunk_id")
            content = rrf_chunk.get("content")
            if not content:
                continue
            # 构建字典
            data = {
                "chunk_id": chunk_id,
                "content": content,
            }
            final_data.append(data)

        # 获取web搜索结果列表，遍历结果封装
        web_search_docs = state.get("web_search_docs")
        for doc in web_search_docs:
            title = doc.get("title")
            url = doc.get("url")
            content = doc.get("snippet")
            if not content:
                continue
            data = {
                "title": title,
                "url": url,
                "content": content,
            }
            final_data.append(data)
        return final_data

    # 把用户问题 + 答案 传递给reranker模型，计算每个答案匹配度分数
    # 返回列表，包含答案 + 匹配度分数 ，根据分数排序
    # user_query 用户问题
    # merged_multi_data：问题对应答案列表
    def reranker_data_score(self,
                            user_query:str,
                            merged_multi_data:List[Dict[str,Any]]
                            ) -> List[Dict[str,Any]]:
        # 获取reranker模型
        reranker_model = get_reranker_model()

        # 构建传递到reranker模块数据
        # [(问题,答案1),(问题,答案2)]
        query_answer = [(user_query, data.get("content"))
                        for data in merged_multi_data ]
        # [0.9, 0.3,  0.11]
        reranker_score = (
            reranker_model.compute_score(sentence_pairs=query_answer))
        print("=="*50)
        print(reranker_score)
        print("==" * 50)

        # 构建返回数据
        #[{ content:11} , { content:22}]  [0.9, 0.3]

        # [
        #   { content:11 ,score:0.9}
        #  { content:22 ,score:0.3}
        # ]
        doc_score = [{**doc,"score":scorce}
                        for doc,scorce in zip(merged_multi_data,reranker_score)
                     ]

        # 对doc_score列表根据分数排序
        sorted_data = sorted(doc_score, key=lambda x: x["score"], reverse=True)
        print("***" * 50)
        print(reranker_score)
        print("***" * 50)
        return sorted_data

    # 断崖检测
    # [
    #   { content:11 ,score:0.9}
    #  { content:22 ,score:0.3}
    # ]
    def cliff_data(self,reranker_doc):
        # 设置上限和下限
        upper = min(10,len(reranker_doc))
        lower = min(3,upper)

        # 根据下限和上限边界遍历
        # 0.9 0.8 0.7  0.1
        cut_pos = upper
        for i in range(lower-1,upper-1):
            # 获取当前分数
            current_score = reranker_doc[i].get("score")
            # 获取下一个分数
            next_score = reranker_doc[i+1].get("score")

            if current_score is None or next_score is None:
                continue

            # 计算差值
            chazhi = current_score - next_score

            # 阈值 差值如果大于等于0.5
            if chazhi >= 0.5:
                cut_pos = i+1
                break

        # [0,3)
        return reranker_doc[:cut_pos]

# if __name__ == "__main__":
#
#     mock_state = {
#         "rewritten_query": "怎么测这块主板的短路问题？",
#         "rrf_chunks": [
#             {"chunk_id": "local_1", "title": "主板维修手册",
#              "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
#             {"chunk_id": "local_2", "title": "闲聊",
#              "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
#         ],
#         "web_search_docs": [
#             {"url": "https://example.com/repair", "title": "短路查修指南",
#              "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
#             {"url": "https://example.com/news", "title": "科技新闻",
#              "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
#         ],
#     }
#
#     node = RerankerSearchNode()
#     result = node.process(mock_state)
#     print(json.dumps(result, indent=2,ensure_ascii=False))