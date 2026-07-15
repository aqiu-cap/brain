
"""
    1 创建graph对象
    2 添加节点（普通节点 和 虚拟节点）
    3 添加边（普通边 和 条件边）
    4 编译graph
    5 执行
"""
from langgraph.constants import END
from langgraph.graph import StateGraph

from knowledge.processor.import_process.nodes.itemname_recognition_node import ItemNameRecognitionNode
from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
from knowledge.processor.query_process.nodes.hyde_search_node import HydeSearchNode
from knowledge.processor.query_process.nodes.item_name_confirm_node import ItemNameConfirmNode
from knowledge.processor.query_process.nodes.reranker_search_node import RerankerSearchNode
from knowledge.processor.query_process.nodes.rrf_search_node import RRFSearchNode
from knowledge.processor.query_process.nodes.vector_search_node import VectorSearchNode
from knowledge.processor.query_process.nodes.web_search_node import WebSearchNode
from knowledge.processor.query_process.state import QueryGraphState

# 条件边路由方法
def router_item_name(state:QueryGraphState)->bool:
    if state.get("answer"):
        return True
    return False

# # 返回编译之后graph对象
def create_query_graph():
    # 1 创建graph对象
    graph = StateGraph(QueryGraphState)
    # 2 添加节点（普通节点 和 虚拟节点）
    nodes = {
        "item_name_confirm":ItemNameConfirmNode(),
        # 创建虚拟节点，多路并行操作
        "multi_search": lambda x: x,
        "search_embedding":VectorSearchNode(),
        "search_embedding_hyde":HydeSearchNode(),
        "web_search_mcp":WebSearchNode(),
        # 创建虚拟节点，把上面三个检索结果合并
        "join": lambda x: {},
        "rrf":RRFSearchNode(),
        "reranker":RerankerSearchNode(),
        "answer_output":AnswerOutputNode(),
    }
    for name,node in nodes.items():
        graph.add_node(name, node)

    # 3 添加边（普通边 和 条件边）
    graph.set_entry_point("item_name_confirm")
    # 添加条件边
    # item_name_confirm 如果有answer 直接进入答案生成节点
    #                    没有answer，执行后面多路检索过程
    graph.add_conditional_edges(
        "item_name_confirm",
        router_item_name,
        {
            False: "multi_search",
            True: "answer_output"
        }
    )
    # 虚拟节点实现：多路并行操作
    graph.add_edge("multi_search", "search_embedding")
    graph.add_edge("multi_search", "search_embedding_hyde")
    graph.add_edge("multi_search", "web_search_mcp")

    # 虚拟节点实现：多路结果在这个节点合并
    graph.add_edge("search_embedding", "join")
    graph.add_edge("search_embedding_hyde", "join")
    graph.add_edge("web_search_mcp", "join")

    # 其他普通边
    graph.add_edge("join", "rrf")
    graph.add_edge("rrf", "reranker")
    graph.add_edge("reranker", "answer_output")
    graph.add_edge("answer_output", END)

    # 4 编译graph
    return graph.compile()

# 获取编译graph对象
query_app = create_query_graph()

# if __name__ == "__main__":
#
#     print("开始测试: 查询流程主图 (main_graph)")
#     mock_state_1 = {
#         "original_query": "我想知道H3C LA2608如何使用？",
#         "session_id": "test_session_main_graph",
#         "task_id": "test_task_001",
#         "is_stream": False,
#     }
#
#     result_1 = query_app.invoke(mock_state_1)
#     print(result_1)