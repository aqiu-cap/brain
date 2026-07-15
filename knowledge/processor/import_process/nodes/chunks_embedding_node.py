import json
from typing import Any, List, Dict
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.bge_client_util import get_bgem3_client

# 切片向量化
class ChunksEmbeddingNode(BaseNode):
    def process(self,state:ImportGraphState)->ImportGraphState:

        self.logger.info("节点6：开始切片向量化")
        # 1 从上一个节点state获取chunks切片列表
        chunks = state.get("chunks")
        if not chunks:
            raise ValidationError("No chunks found")

        # 2 把chunks列表遍历，得到每部分chunk切片数据，向量化
        # 如果chunks列表内容很多，一次性提交bge-m3嵌入模型造成溢出风险，
        # 批量处理  规则：每3段
        # 1 2 3 4 5 6 7 =》 1 2 3    4 5 6  7
        #                  chunks[0:3]      chunks[3:6]
        # range(0,10,3) = [0,0+3) 0 1 2   = [3,3+3) 3 4 5
        # 定义变量
        batch_size = 3
        total = len(chunks)
        final_chunks = []
        # range 获取每次前三段下标
        for start_index in range(0, total, batch_size):
            # chunks切片  chunks[0:3]      chunks[3:6]
            batch_chunks = chunks[start_index:start_index+batch_size]

            # 调用bge-m3嵌入模型向量化
            # 对切片内容向量化，生成两个向量
            # 把两个向量添加每段内容里面
            # 返回列表
            embedding_result = self.execute_batch_embedding(batch_chunks)

            # 放到final_chunks
            final_chunks.extend(embedding_result)
        # 更新state
        state["chunks"] = final_chunks
        return state

    # 调用bge-m3嵌入模型向量化
    # 对切片内容向量化，生成两个向量
    # 把两个向量添加每段内容里面
    # 返回列表
    def execute_batch_embedding(self,
             batch_chunks:List[Dict[str,Any]]):
        final_chunks = []
        for index,chunk in enumerate(batch_chunks):
            # 从每个chunk获取item_name和content
            item_name = chunk.get("item_name")
            content = chunk.get("content")
            embed_content = f"{item_name}\n{content}"
            # 每段内容放到列表
            # ["item_name+content1","item_name+content1+2","item_name+content3"]
            final_chunks.append(embed_content)

        # 获取嵌入模型对象
        bgem3_client = get_bgem3_client()
        vector_result = bgem3_client.encode_documents(final_chunks)

        # vector_result获取每段内容两个向量，把两个向量放到每段内容里面
        for index,chunk in enumerate(batch_chunks):
            # 密集向量 [[第一段密集],[第二段密集],[第三段密集]]
            dense_vector = vector_result['dense'][index].tolist()

            # 稀疏向量 csr结构
            csr = vector_result["sparse"]
            # 指针
            indptr = csr.indptr
            # 获取每段内容位置，切片操作
            start_index = indptr[index]
            end_index = indptr[index+1]

            # tokenid
            token_id = csr.indices[start_index:end_index].tolist()

            # 权重
            data = csr.data[start_index:end_index].tolist()

            # 稀疏向量字典
            sparse_vector = dict(zip(token_id,data))

            # 把密集向量 和 稀疏向量 放到每段内容里面
            chunk["dense_vector"] = dense_vector
            chunk["sparse_vector"] = sparse_vector

        return batch_chunks

if __name__ == "__main__":
    # 获取上一步内容，当前chunks_itemname_0603.json
    # 构建数据
    input_path = r"D:\know\test\chunks_itemname_0626.json"
    with open(input_path,"r",encoding="utf-8") as f:
        total_content = json.load(f)


    state:ImportGraphState = {
        "chunks": total_content.get('chunks')
    }

    chunck_embed_node = ChunksEmbeddingNode()
    result = chunck_embed_node.process(state)

    # 把执行之后result结果输出到json文件里面
    output_path = r"D:\know\test\chunks_embedding_0627.json"
    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=4)
