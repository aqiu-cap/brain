import asyncio
from knowledge.processor.query_process.nodes.web_search_node import WebSearchNode
# ⚠️ 请根据实际目录结构调整下方导入路径
from knowledge.processor.query_process.config import get_config


async def main():
    print("🔍 开始独立测试 WebSearchNode...")

    # 1. 获取全局配置单例（自动读取 .env）
    config = get_config()

    # 验证关键配置是否加载成功
    if not config.mcp_dashscope_base_url or not config.openai_api_key:
        print("❌ 配置缺失！请检查 .env 中的 MCP_DASHSCOPE_BASE_URL 和 OPENAI_API_KEY")
        return

    print(f"✅ 配置加载成功，MCP URL: {config.mcp_dashscope_base_url[:30]}...")

    # 2. 实例化节点并注入配置
    node = WebSearchNode()
    node.config = config

    # 3. 构造最小化 State
    state = {
        "rewritten_query": "今天合肥的天气怎么样？"
    }

    # 4. 调用标准入口 process
    try:
        result = node.process(state)
        docs = result.get("web_search_docs", [])
        print(f"✅ 测试成功！返回 {len(docs)} 条搜索结果:")
        for doc in docs[:3]:
            print(f"   - [{doc.get('title')}] {doc.get('url')}")
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print(f"🚨 捕获到事件循环冲突！这就是线上搜索失败的元凶: {e}")
        else:
            print(f"❌ 其他 RuntimeError: {e}")
    except Exception as e:
        print(f"❌ 测试失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())