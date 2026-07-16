import asyncio
import json
from typing import Dict, Any
import concurrent.futures

from agents.mcp import MCPServerStreamableHttp

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState

# web搜索节点
class WebSearchNode(BaseNode):
    def process(self,state:QueryGraphState)->QueryGraphState:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            # 没有运行中的事件循环 → 直接新建一个
            return asyncio.run(self._async_process(state))
        else:
            # 已有运行中的事件循环（FastAPI async handler 场景）
            # → 开独立线程执行，避免阻塞主事件循环
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self._async_process(state)
                ).result()

    # 异步方法开始调用过程
    async def _async_process(self,state:QueryGraphState)->QueryGraphState:
        # 从state获取用户重写问题
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise Exception("rewritten_query is None")

        # 调用异步方法 mcp工具调用
        mcp_result = await self.execute_mcp_search(rewritten_query)
        if not mcp_result:
            return {"web_search_docs": []}
        return {"web_search_docs": mcp_result}
    # 异步方法 mcp工具调用
    # 异步方法 mcp工具调用
    async def execute_mcp_search(self,rewritten_query:str):
        # streamablehttp和服务端建立连接
        mcp_client = None
        # 构建headers数据，鉴权操作
        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json"
        }
        try:
            # streamablehttp创建客户端对象，建立和服务端连接
            mcp_client = MCPServerStreamableHttp(
                name="通用搜索",
                params={"url": self.config.mcp_dashscope_base_url,
                        "headers": headers,
                        },
                cache_tools_list=True,
            )

            # 建立和服务端连接
            await mcp_client.connect()

            # 调用工具方法
            execute_tool_result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={
                    "query": rewritten_query,
                    "count":3
                }
            )
            if not execute_tool_result:
                return []

            # execute_tool_result获取需要数据，构建新列表返回
            content_text:str = execute_tool_result.content[0].text
            # content_text转换字典 反序列化
            data:Dict[str,Any] = json.loads(content_text)

            search_result = []
            # data字典有固定key名称 pages
            pages = data.get("pages")
            # pages列表
            for page in pages:
                snippet = page.get("snippet")
                title = page.get("title")
                url = page.get("url")
                # 封装到search_result
                search_result.append(
                    {
                        "snippet": snippet,
                        "title": title,
                        "url": url
                    }
                )
            return search_result

        except Exception as e:
            raise e
        finally:
            # 连接对象释放
            if mcp_client:
                await mcp_client.cleanup()
