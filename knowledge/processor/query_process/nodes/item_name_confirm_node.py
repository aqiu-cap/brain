import json
import re
from typing import Dict, List, Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.bge_client_util import get_bgem3_client, generate_vector_data
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import get_milvus_client, create_vector_search_request, \
    execute_bybrid_search_query
from knowledge.utils.mongo_client_util import get_recent_message


# 内部类1：LLM操作类
class ItemNameLLM:
    # original_query 用户输入原始问题
    # chat_history 前10条历史会话记录列表
    ## 返回字典
    def call_llm_item_name(self,original_query:str,
                           chat_history:List)->Dict:

        # chat_history列表处理，获取历史会话信息转换字符串
        history = ""
        for message in chat_history:
            role = message.get("role")
            text = message.get("text")
            history += f"{role}:{text}\n"

        # 构建提示词
        system_prompt = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"
        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            query=original_query,
            history_text=history
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        # 提交LLM调用
        llm_client = get_llm_client(response_format=True)
        response = llm_client.invoke(messages)
        # getattr(response, "content")
        # { }
        result = response.content.strip()
        # 对返回结果清洗，清理不需要数据或者字符 ，json代码块： ```json    ```
        final_result = self.clean_content(result)
        return final_result

    # 对返回结果清洗，清理不需要数据或者字符 ，json代码块： ```json    ```
    def clean_content(self,result)->Dict[str,Any]:
        # ```json
        # re.sub(正则, 替换之后内容, 从哪里)
        cleand = re.sub(r"^```(?:json)?\s*","",result)
        # ```
        final_content = re.sub(r"\s*```$","",cleand)

        # final_content字符串类型 变成字典类型
        cleaned_result:Dict[str,Any] = json.loads(final_content)

        # ori_item_names列表 ["商品A", "商品B"]
        ori_item_names = cleaned_result.get('item_names')
        cleaned_item_names = [ori_item for ori_item in ori_item_names if ori_item.strip()]
        return {
            "item_names": cleaned_item_names,
            "rewritten_query": cleaned_result.get("rewritten_query")
        }

# 内部类2：查询向量数据库操作类
class ItemNameVector:
    """
        1 把传递item_names 生成两个向量
        2 把生成两个向量构建向量查询条件
        3 执行混合查询，得到结果和分数
        4 按照结果分数排序 从高到低排序
        5 评分对齐（过滤）
        * 分数阈值 0.7
        * 分数大于0.7 放到 confirmed 列表 放一条大于0.7数据
        ** 如果分数大于0.7 有多条，获取分数最高
        * 分数0.6 - 0.7 之间 放到 options列表
        ** 可能有一条，可能有多条 ，约定最多放3条
        # 最终返回两个列表 confirmed  options
    """
    # ["商品名1", "商品名2"]
    def execute_item_name_search(self,
              item_names:List[str])->Tuple[List[str],List[str]]:
        # 1 item_names查询，得到结果和分数
        search_result:List[Dict[str,Any]] = self.vector_search(item_names)

        # 2 评分对齐  0.7  0.6-0.7
        confirmed,options = self.item_name_score(search_result)
        return confirmed,options

    # 1 item_names查询，得到结果和分数
    def vector_search(self,item_names:List[str])->List[Dict[str,Any]]:
        # 获取milvus 和 bgem3对象
        milvus_client = get_milvus_client()
        bgem3_client = get_bgem3_client()
        # 封装最终数据
        search_result = []

        # 把传递item_names生成两个向量
        # return {  # [[111],[333]] => [111]     [333]
        #     "dense": [dense.tolist() for dense in embedding_result['dense']],
        #     "sparse": final_sparse_vector,
        # }
        embedding_result = generate_vector_data(bgem3_client,item_names)

        """
            ["商品名1", "商品名2"]
            
            {
                "dense": [[商品名1密集向量数据], [商品名2密集向量数据]]
                "sparse":[{商品名1稀疏向量数据} ,{商品名2稀疏向量数据}] 
            }
        """
        # 遍历item_names:List[str]
        for index,item_name in enumerate(item_names):
            # 根据遍历index索引，从 embedding_result把每个商品名对应密集和稀疏向量数据获取
            dense_vector = embedding_result["dense"][index]
            sparse_vector = embedding_result["sparse"][index]

            # 构建向量查询条件
            bybrid_search_request = create_vector_search_request(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
            )

            # 执行混合查询
            hybrid_search_result = execute_bybrid_search_query(
                milvus_client=milvus_client,
                collection_name="kb_item_names_v2",
                search_request=bybrid_search_request,
                norm_score=True
            )

            # 混合查询结果重新组装，从结果获取item_name 和 分数
            """
                data: [
                            [
                                {
                                    'pk': '467259340667782119',
                                    'distance': 0.7221629619598389,
                                    'entity': {
                                        'item_name': 'H3C LA2608 室内无线网关'
                                    }
                                }
                            ]
                        ]
                """
            item_name_score_result = {
                "extracted_name":item_name,
                "matches":[
                    {
                        "item_name": h['entity']['item_name'],
                        "score": h['distance']
                    }
                    for h in (hybrid_search_result[0]
                               if hybrid_search_result else [])
                ]
             }

            # 每个item_name数据放到最终列表里面
            search_result.append(item_name_score_result)
        return search_result

    # 2 评分对齐  0.7  0.6-0.7
    # # 最终返回两个列表 confirmed  options
    def item_name_score(self,
                search_result:List[Dict[str,Any]])->Tuple[List[str],List[str]]:
        """
        [
         {
            'extracted_name': 'H3C LA2608',
            'matches': [
            {
                'item_name': 'H3C LA2608 室内无线网关',
                'score': 0.7221629619598389
            }
            ]
         },
         {
            'extracted_name': 'H3C LA2608',
            'matches': [
            {
                'item_name': 'H3C LA2608 室内无线网关',
                'score': 0.7221629619598389
            }
            ]
         }
        ]
        """
        # 创建两个列表
        confirmed = [] # 分数大于等于0.7
        options = []   # 分数在0.6 - 0.7之间

        # 遍历search_result:List
        for item_name_search_result in search_result:
            # 根据每部分里面score排序，从高到低
            matches = sorted(
                item_name_search_result.get("matches"),
                key=lambda x: x["score"],
                reverse=True)

            # 根据分数过滤
            high = [m for m in matches if m.get("score")>=0.7]
            # 如果matches存在大于等于0.7数据，把数据里面item_name放到confirmed
            # 问题：大于等于0.7数据可能有多个，把第一个放到列表
            if high:
                high_item_name = high[0]["item_name"]
                if high_item_name not in confirmed:
                    confirmed.append(high_item_name)
            else:
                # 获取分数在0.6 到 0.7之间数据
                mid = [m for m in matches if m.get("score")>=0.6]
                # 在0.6 到 0.7之间数据可能没有，可能多个，最多放前3个
                if mid:
                    for m in mid[:3]:
                        mid_item_name = m.get("item_name")
                        if mid_item_name not in options:
                            options.append(mid_item_name)
        return confirmed,options

# 商品名确认模块
"""
    # 1 获取用户输入原始问题
    # 2 根据session_id获取用户前10条会话记录
    # 3 根据用户问题+会话记录 构建提示词
    # 4 把构建提示词提交LLM，提取商品名 和 用户问题重写
    # 5 根据提取商品名构建向量查询条件，进行混合检索，得到结果和匹配度分数
    # 6 对分数排序和评分对齐
    # 7 更新state相关数据
"""
class ItemNameConfirmNode(BaseNode):
    # 初始化方法
    def __init__(self):
        super().__init__()
        self._item_name_llm = ItemNameLLM()
        self._item_name_vector = ItemNameVector()

    # 流程方法
    def process(self,state:QueryGraphState)->QueryGraphState:
        # 1 获取用户输入原始问题
        original_query = state.get("original_query")

        # 2 根据session_id获取用户前10条会话记录
        session_id = state.get("session_id")
        # 调用MongoDB工具类的方法
        # chat_history 列表
        chat_history = get_recent_message(session_id)

        # 调用内部类方法构建提示词，调用llm返回结果
        llm_result = self._item_name_llm.call_llm_item_name(
                                  original_query, chat_history)

        # 返回格式字典格式
        # {
        #     "item_names": ["商品A", "商品B"],
        #     "rewritten_query": "关于商品A和商品B，..."
        # }
        # item_names多个商品名列表
        item_names = llm_result.get("item_names")
        # 重写之后用户的问题
        rewritten_query = llm_result.get("rewritten_query")

        # 根据提取商品名构建向量查询条件，进行混合检索，得到结果和匹配度分数
        if item_names:
            # 调用内部类的方法 构建向量条件，混合查询，返回结果和分数
            # 直接传递item_names
            # 返回结果：按照需求决定的
            # 返回两个列表  [] []
            # 约定  第一个列表 分数大于0.7结果  最终一个结果，如果多个获取分数最高
            #      第二个列表 分数大于0.6 小于0.7 结果  可能有多个，让用户自己选择
            confirmed,options = self._item_name_vector.execute_item_name_search(
                                        item_names)
        else:
            confirmed, options = [],[]

        # 更新state数据
        self.update_state(state,confirmed, options,rewritten_query)
        state['history'] = chat_history
        return state

    # 更新state
    def update_state(self,state,confirmed, options,rewritten_query):
        # confirmed有数据，有分数大于0.7数据，
        # 用户问题重写成功，更新state数据，执行后面节点
        if confirmed:
            state["rewritten_query"] = rewritten_query
            state["item_names"] = confirmed
        # options有数据，分数在0.6 到 0.7 之间数据
        # options数据返回给用户，用户选择到底是哪个问题
        elif options:
            state["answer"] = f"请选择问题：{','.join(options)}"
        else:
            state["answer"] = "当前问题无法识别...."


# if __name__ == "__main__":
#     state: QueryGraphState = {
#         "original_query": "我想知道H3C LA2608如何使用？"
#         # "original_query": "我的手机如何使用?"
#     }
#
#     node = ItemNameConfirmNode()
#     res = node.process(state)
#
#     print(res)