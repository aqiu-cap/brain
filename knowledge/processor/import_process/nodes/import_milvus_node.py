import json
from typing import List, Dict, Any

from pymilvus import CollectionSchema, MilvusClient, DataType
from pymilvus.milvus_client import IndexParams
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.milvus_client_util import get_milvus_client


# 内部类1：创建milvus里面集合约束
class _MilvusSchemaBuiler:
    # 静态方法：集合约束
    @staticmethod
    def build_schema(milvus_client:MilvusClient)->CollectionSchema:
        schema = milvus_client.create_schema()
        # 主键
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            auto_id=True,
            max_length=1000
        )
        schema.add_field(
            field_name="title",
            datatype=DataType.VARCHAR,
            max_length=1000,
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=1000,
        )
        schema.add_field(
            field_name="parent_title",
            datatype=DataType.VARCHAR,
            max_length=1000,
        )
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=1000,
        )
        # 向量字段
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024  # 维度
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        return schema

# 内部类2：创建milvus里面集合字段索引
class _MilvusIndexBuilder:
    @staticmethod
    def build_index(milvus_client:MilvusClient)-> IndexParams:
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        return index_params

# 内部类3：添加数据到milvus方法
class _MilvusSaveBuilder:

    # 创建_MilvusSaveBuilder对象，执行 init方法
    # 数据初始化，传递两个参数
    # milvus连接对象
    # 集合名称
    def __init__(self,
                 milvus_client:MilvusClient,
                 collection_name:str):
        self.milvus_client = milvus_client
        self.collection_name = collection_name

    # 把传递过来chunks列表里面每段内容添加向量数据库里面
    # 最终返回chunks里面--每部分内容添加chunk_id(这段内容在向量数据库主键值)
    def insert(self,chunks:List[Dict[str,Any]]
                    )->List[Dict[str,Any]]:

        inserted_result = self.milvus_client.insert(
            collection_name=self.collection_name,
            data=chunks
        )
        # print("==" *50)
        # print(f"添加之后返回结果：{inserted_result}")
        # print("==" * 50)
        # 调用insert方法之后，通过返回结果得到添加之后主键值
        # 固定名称 ids ，通过ids获取主键值列表
        ids = inserted_result.get("ids")
        # ids每个主键值获取，更新 对应chunks列表里面
        self.update_chunks_ids(ids,chunks)
        return chunks

    def update_chunks_ids(self,ids,chunks):
        # chunks  [{..},{...}]
        # ids     [ 1  ,  2 ]
        for chunk,id in zip(chunks,ids):
            chunk["chunk_id"] = id

# 主类
# 把切片向量化数据添加milvus
class ImportMilvusNode(BaseNode):
    def process(self,state:ImportGraphState)->ImportGraphState:
        self.logger.info("节点7：开始导入切片数据到milvus")
        # 1 获取上一步返回chunks列表数据
        chunks = state.get("chunks")
        if not chunks:
            raise ValidationError("chunks null")

        # 2 判断milvus是否存在集合
        milvus_client = get_milvus_client()
        collection_name = self.config.chunks_collection
        # 调用方法实现
        self.is_has_collection(milvus_client, collection_name)

        # 3 调用内部类的方法实现添加
        milvus_save_obj = _MilvusSaveBuilder(milvus_client=milvus_client,
                                             collection_name=collection_name)
        final_chunks = milvus_save_obj.insert(chunks=chunks)

        # 4 返回列表更新state
        state["chunks"] = final_chunks
        return state

    # 2 判断milvus是否存在集合
    def is_has_collection(self,milvus_client:MilvusClient,
                          collection_name:str):
        if not milvus_client.has_collection(collection_name):
            # 创建集合约束
            schema = _MilvusSchemaBuiler.build_schema(milvus_client)
            # 创建索引
            index_params = _MilvusIndexBuilder.build_index(milvus_client)
            # 调用方法
            milvus_client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )

if __name__ == '__main__':
    input_path = r"D:\know\md\chunks_embedding_0627.json"
    with open(input_path,"r",encoding="utf-8") as f:
        file_content = json.load(f)

    state:ImportGraphState = {
        "chunks": file_content.get("chunks")
    }

    # 调用
    import_milvus = ImportMilvusNode()
    result = import_milvus.process(state)

    # 调用返回结果写入到新json文件里面
    output_path = r"D:\know\md\chunks_import_milvus_0627.json"
    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=4)