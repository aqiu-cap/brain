import json
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import MilvusClient, DataType

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.upload.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.bge_client_util import get_bgem3_client
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import get_milvus_client


# 商品名识别模块
class ItemNameRecognitionNode(BaseNode):
    def process(self,state:ImportGraphState)->ImportGraphState:

        self.logger.info("节点5：开始提取商品名")
        #1 参数校验 切分列表数据
        chunks,file_title = self.get_operator_data(state)
        self.logger.info("第一步 参数校验...")
        self.logger.info(file_title)

        #2 切分列表获取前5段内容
        # 返回切片前5段数据字符串 prompt_data
        prompt_data = self.get_chunks_prompt_data(chunks)
        self.logger.info("第二步 切分列表获取前5段内容...")
        # self.logger.info(prompt_data)

        #3 获取切分列表前5段构建提示词，提交LLM，返回提取商品名
        item_name = self.call_llm_item_name(prompt_data,file_title)
        self.logger.info("第三步 调用llm提取商品名...")
        self.logger.info(item_name)

        #4 把提取商品名嵌入（生成向量：密集 和 稀疏向量）
        dense_vector,sparse_vector = self.get_dense_sparse_item_name(item_name)
        self.logger.info("第四步 对商品名生成两个向量...")
        # self.logger.info(dense_vector)
        # self.logger.info(sparse_vector)

        #5 把生成商品名向量和其他字段值存储milvus向量数据库里面
        self.save_milvus_data(dense_vector,sparse_vector,
                                 item_name,file_title)
        self.logger.info("第五步 存储到milvus向量数据库...")

        #6 更新state状态数据
        self.update(state,chunks,item_name)
        return state

    # 6 更新state状态数据
    def update(self,state:ImportGraphState,
               chunks:List[Dict],item_name:str):
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["item_name"] = item_name

    # 1 #1 参数校验 切分列表数据
    def get_operator_data(self,state:ImportGraphState):
        # 切分列表数据
        chunks = state.get("chunks")
        if not chunks:
            raise ValidationError("切分列表数据不存在")
        file_title = state.get("file_title")
        if not file_title:
            raise ValidationError("file_title不存在")
        return chunks,file_title

    # 2 切分列表获取前5段内容
    # chunks列表获取前5段数据
    # 把前5段数据拼接字符串，字符数量限制 2000
    def get_chunks_prompt_data(self,chunks:List[Dict])->str:
        result = []
        total = 0               # [0,5) => 0 1 2 3 4
        for index,chunk in enumerate(chunks[:5]):
            # chunk字典
            content = chunk.get("content")
            chunk_content = f"数据-{index+1}-{content}"
            #字符数量计算
            total += len(chunk_content)
            # result里面
            result.append(chunk_content)
            # 字符限制
            if total > self.config.max_content_length:
                break
        return "\n".join(result)

    # 3 获取切分列表前5段构建提示词，提交LLM，返回提取商品名
    def call_llm_item_name(self,prompt_data:str,file_title:str)->str:
        # 根据传递prompt_data字符串构建提示词
        user_prompt_data = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
            file_title=file_title,
            context=prompt_data
        )
        messages = [
            SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt_data),
        ]

        # messages1 = [
        #     {
        #         "role": "user",
        #         "content": prompt_data,
        #     },
        #     {
        #         "role": "system",
        #         "content": prompt_data,
        #     },
        # ]

        # 把提示词提交llm，生成商品名
        # 获取llm连接对象
        client = get_llm_client()
        # 调用llm，传递提示词，返回结果
        response = client.invoke(messages)

        # 在返回response对象，有属性content，content属性最终内容
        # item_name = response.content
        # getattr从一个对象里面获取某个属性值
        item_name = getattr(response,"content","")
        # 如果item_name
        if not item_name:
            self.logger.warn("当前item_name提取为空，使用默认值")
            # file_title作为默认商品名
            return file_title
        return item_name

    # 根据item_name 使用嵌入模型 生成 密集 和稀疏向量
    def get_dense_sparse_item_name(self,item_name:str):
        # 获取bge-m3模型对象
        bgem3_client = get_bgem3_client()
        # bgem3_client方法生成两个向量 ["商品名称"]
        embedding_result = bgem3_client.encode_documents([item_name])
        # 从embedding_result嵌入结果里面获取密集和稀疏向量
        # 获取密集向量 [[0.1,0.3]]
        dense_vector = embedding_result['dense'][0].tolist()
        # 获取稀疏向量
        sp = embedding_result['sparse']
        # 稀疏向量返回结果有三部分：指针 、token id编号、编号对应权重
        #sp.indptr.tolist() # 指针
        #tokenid编号 => 权重
        sparse_vector = dict(zip(sp.indices.tolist(),sp.data.tolist()))
        # 返回
        return dense_vector,sparse_vector

    # 把这些数据包含向量字段数据 + 标量字段数据添加milvus数据库
    def save_milvus_data(self,dense_vector,sparse_vector,
                         item_name,file_title):
        # 1 获取milvus_client
        milvus_client = get_milvus_client()
        # 集合名称
        collection_name = self.config.item_name_collection
        # 2 判断如果不存在collection，创建
        # kb_chunks_v2
        if not milvus_client.has_collection(collection_name=collection_name):
            self.create_milvus_collection(milvus_client,collection_name)

        # 3 调用milvus_client方法添加数据到milvus
        data = {
            "file_title":file_title,
            "item_name":item_name,
            "dense_vector":dense_vector,
            "sparse_vector":sparse_vector
        }
        result = milvus_client.insert(
            collection_name=collection_name,
            data=[data]
        )
        self.logger.info(f"添加milvus数据完成：{result}")

    # 创建milvus的集合（表）
    def create_milvus_collection(self,milvus_client:MilvusClient,
                                 collection_name:str)->None:
        #1 集合约束：字段名称对应类型等
        # 创建CollectionSchema对象
        schema = milvus_client.create_schema()
        # 设置字段和类型等
        # 设置主键
        schema.add_field(
            field_name="pk", # 字段名称
            datatype=DataType.VARCHAR, # 字段类型
            is_primary=True, # 是否主键 true是主键
            auto_id=True,    # 自动生成主键值
            max_length=1000  # 字段最大长度
        )
        # 设置标量字段 1-65535
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=1000
        )
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=1000,
            # nullable=True # nullable允许字段值为空
        )
        # 设置向量字段
        # 密集向量字段
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024  # 维度
        )
        # 稀疏向量字段
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        #2 字段索引：向量字段索引信息
        # 创建索引对象
        index_params = milvus_client.prepare_index_params()
        # 设置 dense_vector
        index_params.add_index(
            field_name="dense_vector", # 字段名称
            index_name="dense_vector_index", # 索引名称
            index_type="AUTOINDEX",  # milvus自己选择
            metric_type="COSINE"  # 余弦相似度匹配
        )

        index_params.add_index(
            field_name="sparse_vector",  # 字段名称
            index_name="sparse_vector_index",  # 索引名称
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # create_collection方法有三个参数
        # 1 集合名称
        # 2 集合约束：字段名称对应类型等
        # 3 字段索引：向量字段索引信息
        milvus_client.create_collection(
            collection_name=collection_name, # 集合名称
            schema=schema,  # 约束
            index_params=index_params  # 索引
        )

# if __name__ == "__main__":
#     setup_logging()
#     # 上一步文档切分之后json文件路径
#     chunk_json_path = r"D:\know\test\chunks.json"
#     with open(chunk_json_path,"r",encoding="utf-8") as f:
#         chunks_content = json.load(f)
#
#     # file_title,chunks
#     state = {
#         "file_title":"123",
#         "chunks":chunks_content
#     }
#     node = ItemNameRecognitionNode()
#     result = node.process(state)
#
#     # 把执行之后result结果输出到json文件里面
#     output_path = r"D:\know\test\chunks_itemname_0626.json"
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, ensure_ascii=False, indent=4)
#     print(result)