import os

from dotenv import load_dotenv
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

load_dotenv()
def get_milvus_client():
    try:
        milvus_client = MilvusClient(
            uri=os.getenv("MILVUS_URL"),
        )
        return milvus_client
    except Exception as e:
        raise e

# 构建向量条件方法
"""
    * 商品名确认首先通过LLM提取商品名
    * 拿着商品名查询向量数据库得到结果  无线路由器
    * 构建商品名向量条件 密集向量 和 稀疏向量条件
    ** 类似于mysql   where a=? and b=?
"""
# 第一个和第一个参数 商品名生成密集和稀疏向量数据  dense_vector,sparse_vector
# 第三个和第四个参数 索引类型值 密集向量默认值 CONSINE  稀疏向量默认值 IP
# 第五个参数 条件表达
# 第六个参数 单路返回结果数量  密集和稀疏向量结果最终返回多少个  topK
def create_vector_search_request(dense_vector,
                                 sparse_vector,
                                 dense_params=None,
                                 sparse_params=None,
                                 expr=None,limit=5):
    # 判断第三个和第四个参数
    # 索引类型值 密集向量默认值 COSINE  稀疏向量默认值 IP
    if dense_params is None:
        dense_params = {"metric_type": "COSINE",}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP",}

    # 构建稠密向量条件
    dense_request = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit,
    )

    # 构建稀疏向量条件
    sparse_request = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit,
    )
    return [dense_request, sparse_request]

# 向量数据库混合查询方法
# 第一个参数：milvus_client milvus连接对象
# 第二个参数：collection_name 向量数据库名称
# 第三个参数：构建向量条件
# 第四个参数：权重比例两个值  (0.5,0.5)
# 第五个参数：是否归一化操作
# 第六个参数：结果数量设置
def execute_bybrid_search_query(milvus_client: MilvusClient,
                                collection_name,
                                search_request,
                                reranker_weight=(0.5,0.5),
                                norm_score=False,
                                limit=5,
                                output_fields=None,
                                search_param=None):

    # 创建权重融合排序器
    ## 把稠密向量数据 和 稀疏数量查询数据，根据权重融合，计算分数
    reranker = WeightedRanker(
        reranker_weight[0],reranker_weight[1],
        norm_score=norm_score,
    )

    if output_fields is None:
        output_fields = ["item_name"]

    res = milvus_client.hybrid_search(
        collection_name=collection_name,
        reqs=search_request,
        ranker=reranker,
        limit=limit,
        output_fields=output_fields,
        search_params=search_param,
    )
    return res