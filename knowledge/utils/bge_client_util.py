import os
from typing import List

from dotenv import load_dotenv
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

load_dotenv()
def get_bgem3_client():
    try:
        bge_m3_client = BGEM3EmbeddingFunction(
            model_name=os.getenv("BGE_M3_PATH"),
            device=os.getenv("BGE_DEVICE"),
            use_fp16=False,
        )
        return bge_m3_client
    except Exception as e:
        raise e

# 根据传递数据生成稠密和稀疏向量
# ["商品名1", "商品名2"]
def generate_vector_data(bge_m3_client: BGEM3EmbeddingFunction,
                         documents:List[str]):
    # 使用嵌入模型对象，调用方法传入列表，生成数据
    embedding_result = bge_m3_client.encode_documents(documents)

    # 获取embedding_result里面稀疏向量数据
    final_sparse_vector = []

    for index in range(len(documents)):
        # 获取稀疏向量 csr结构
        csr = embedding_result['sparse']

        # csr指针
        ind_ptr = csr.indptr

        # 获取开始和结束位置
        start_ind = ind_ptr[index]
        end_ind = ind_ptr[index+1]

        # 获取稀疏向量 tokenid  [1,2,3]
        token_id = csr.indices[start_ind:end_ind].tolist()
        # 获取稀疏向量 权重     [0.1,0.9,0.7]
        weight = csr.data[start_ind:end_ind].tolist()

        # tokenid + 权重构建字典
        sparse_vector = dict(zip(token_id, weight))

        # 放到最终列表里面
        final_sparse_vector.append(sparse_vector)

    return {   # [[111],[333]] => [111]     [333]
        "dense": [dense.tolist() for dense in embedding_result['dense']],
        "sparse": final_sparse_vector,
    }




