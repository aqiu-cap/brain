import json

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from knowledge.front.service.task_service import TaskService
from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.chunks_embedding_node import ChunksEmbeddingNode
from knowledge.processor.import_process.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.import_milvus_node import ImportMilvusNode
from knowledge.processor.import_process.nodes.itemname_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.md_img_node import MdImageNode
from knowledge.processor.import_process.nodes.pdf_md_node import PdfToMdNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state


# 1 创建graph对象
# 添加节点，添加边，graph对象编译
# 返回编译之后graph对象
def create_graph_import():
    # 创建graph对象
    graph = StateGraph(ImportGraphState)

    # 添加节点
    nodes = {
        "entry_node":EntryNode(),
        "pdf_to_md":PdfToMdNode(),
        "md_img_node": MdImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_rec_node": ItemNameRecognitionNode(),
        "bge_embedding_node": ChunksEmbeddingNode(),
        "import_milvus_node": ImportMilvusNode()
    }
    # graph.add_node("entry_node",EntryNode())
    for key,value in nodes.items():
        graph.add_node(key, value)

    # 设置入口节点
    graph.set_entry_point("entry_node")

    # 添加边（普通边 和  条件边）
    # 添加条件边
    # add_conditional_edges方法有三个参数
    # 第一个参数：从哪个节点开始
    # 第二个参数：路由方法，判断逻辑，约定如果pdf类型返回pdf  md类型返回md
    # 第三个参数：根据路由方法返回结果配置到不同节点
    graph.add_conditional_edges(
        "entry_node",
        import_router,
        {
            "pdf": "pdf_to_md",
            "md":  "md_img_node",
            END: END
        }
    )

    # 普通边
    # graph.add_edge("entry_node", "pdf_to_md")

    graph.add_edge("pdf_to_md", "md_img_node")

    graph.add_edge("md_img_node", "document_split_node")
    graph.add_edge("document_split_node", "item_name_rec_node")
    graph.add_edge("item_name_rec_node", "bge_embedding_node")
    graph.add_edge("bge_embedding_node", "import_milvus_node")
    graph.add_edge("import_milvus_node", END)

    # 编译，返回
    graph_compile = graph.compile()
    return graph_compile

# 条件边路由方法
def import_router(state: ImportGraphState):
    if state.get("is_pdf_read_enabled"):
        return "pdf"
    if state.get("is_md_read_enabled"):
        return "md"
    return END


# 2 获取上一个方法编译graph对象，执行
# invoke 或者stream方法
# task_id: 任务id，前端通过task_id获取当前7个节点执行进度
# file_dir:            c:\a\b
# import_file_path :  c:\a\b\test.pdf
def run_graph_import(task_id:str, import_file_path:str, file_dir:str):
    # 构建字典数据
    state = {
        "task_id": task_id,
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    # 获取上一个方法返回编译graph对象
    graph = create_graph_import()

    # 解包
    init_state = create_default_state(**state)

    final_state = None
    # 执行
    # res = graph.invoke(state)
    # stream流式输出，返回 Iterator[dict[str, Any] | Any]
    ## 把 Iterator[dict[str, Any] | Any]遍历
    ### 遍历之后得到每个DICT字典
    #### 字典key是节点名称  value是节点执行传递state状态数据
    ts = TaskService()
    for event in graph.stream(init_state):
        for node_name,state_data in event.items():
            print(f"运行节点：{node_name}")
            print(f"传输数据：{state_data}")
            #记录当前节点信息，前端显示
            ts.mark_node_running(task_id, node_name)
            ts.mark_node_done(task_id,node_name)

            final_state=state_data
    return final_state

# if __name__ == "__main__":
#     setup_logging()
#     import_file_path = r"D:\know\123\auto\123.md"
#     file_dir = r"D:\know\test"
#
#     final_res = run_graph_import(
#         task_id=1,
#         import_file_path=import_file_path,
#         file_dir=file_dir,
#     )
#
#     print(json.dumps(final_res,ensure_ascii=False, indent=4))

