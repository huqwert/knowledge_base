# processor/query_process/nodes/node_web_search_mcp.py
"""
    这个节点 负责调用 百炼 MCP (Model Context Protocol) 联网搜索服务 ，获取互联网上的实时信息。
    它通过 MCP 协议异步调用百炼联网搜索接口，将用户的查询转化为实时的、结构化的网络搜索结果（包含标题、链接和摘要）。
"""
import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from processor.config.config import mcp_config
from processor.query_process.base import NodeBase
from processor.query_process.state import QueryGraphState
from processor.tool.logger import logger
from processor.utils.mongo_history_utils import format_json


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) :
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        try:
            query = state.get("rewritten_query")
            result = asyncio.run(self._mcp_call(query))
            logger.info(result)

            pages = json.loads(result.content[0].text).get("pages",[])
            logger.info(f"【{self.name}】pages: {pages}")

            # docs = []
            # # 每条：{title, url, snippet}
            # for item in pages:
            #     snippet = item.get("snippet", "").strip()
            #     title = item.get("title", "").strip()
            #     url = item.get("url", "").strip()
            #     docs.append({"title": title, "url": url, "snippet": snippet})

            docs = [
                {
                    "title": item.get("title", "").strip(),
                    "url": item.get("url", "").strip(),
                    "snippet": item.get("snippet", "").strip()
                } for item in pages
            ]

            return {
                "web_search_docs": docs
            }
        except Exception as e:
            logger.exception(f"MCP搜索失败: {e}")
            return {"web_search_docs": []}

    async def _mcp_call(self,query,count: int=5)-> None:

        async with MCPServerStreamableHttp(
                name="search_mcp",
                params={
                    "url": mcp_config.mcp_base_url,
                    "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                    "timeout": 20,  # 每次 HTTP 请求的超时时间
                },

                cache_tools_list=True,
                max_retry_attempts=3,
                # 它控制的是 MCP 协议层面等待对方回消息的最长时间，
                # 包括握手、列工具、调用工具等所有操作
                client_session_timeout_seconds=30
        ) as server:
            result = await server.call_tool(
                tool_name="bailian_web_search",
                arguments={
                    "query": query,
                    "count":count
                }
            )
            return result


