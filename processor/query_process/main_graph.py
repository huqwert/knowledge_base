from langgraph.constants import START, END
from langgraph.graph import StateGraph

from processor.query_process.nodes.node_answer_output import NodeAnswerOutput
from processor.query_process.nodes.node_rerank import NodeRerank
from processor.query_process.nodes.node_rrf import NodeRrf
from processor.query_process.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_process.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_process.nodes.node_web_search_mcp import NodeWebSearchMcp
from processor.query_process.nodes.node_item_name_confirm import NodeItemNameConfirm
from processor.query_process.state import QueryGraphState
from processor.tool.logger import logger


class KBQueryWorkflow:
    """
    知识库查询工作流类
    封装LangGraph工作流的构建、编译、执行逻辑，支持自定义配置和多实例运行
    """

    def __init__(self):
        """初始化工作流：创建状态图、注册节点、定义路由规则"""
        # 1. 初始化LangGraph状态图
        self.workflow = StateGraph(QueryGraphState)
        # 2. 初始化所有业务节点（实例属性，支持多实例隔离）
        self._init_nodes()
        # 3. 注册节点到工作流
        self._register_nodes()
        # 4. 设置入口和路由规则
        self._setup_routes()
        # 5. 编译工作流（懒加载，首次执行时编译）
        self._compiled_app = None

    def _init_nodes(self):

        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()

    def _register_nodes(self):
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm)
        self.workflow.add_node("node_multi_search",lambda x : x)
        self.workflow.add_node("node_search_embedding", self.node_search_embedding)
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde)
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp)
        self.workflow.add_node("node_json", lambda x : x)
        self.workflow.add_node("node_rrf", self.node_rrf)
        self.workflow.add_node("node_rerank", self.node_rerank)
        self.workflow.add_node("node_answer_output", self.node_answer_output)

    def _route_after_item_name_confirm(self, state: QueryGraphState) ->str:
        """
            这主要发生在 node_item_name_confirm 节点无法直接确定唯一的商品型号，从而需要“反问用户”或“拒绝回答”的场景。
            具体来说，有以下两种情况会导致 state 中直接出现 answer ，从而跳过后续的检索流程，直接输出：
            1. 多选一（反问用户） ：
            - 场景 ：用户问得太模糊（比如“华为P60”），系统发现数据库里有“华为P60 128G”和“华为P60 Art”两个型号，不足以直接确认。
            - 处理 ：节点会生成一条反问句作为 answer ，例如：“您是想问以下哪个产品：华为P60 128G、华为P60 Art？请明确一下型号。”
            - 结果 ：此时不需要去检索文档了，直接把这句话发给用户让他选。

            2. 查无此人（拒绝回答） ：
            - 场景 ：用户问了一个系统里压根没有的商品（比如“小米15”，库里只有华为的数据）。
            - 处理 ：节点会生成一条拒绝句作为 answer ，例如：“抱歉，未找到相关产品，请提供准确型号以便我为您查询。”
            - 结果 ：同样不需要后续检索，直接结束流程。
        """
        # 主体名称确认后的条件路由函数
        if state.get("answer"):
            return "node_answer_output"

        else:
            # 否则继续搜索流程
            return "node_multi_search"

    def _setup_routes(self):
        """设置工作流路由规则"""
        # 1、设置入口节点
        self.workflow.add_edge(START, "node_item_name_confirm")
        # self.workflow.set_entry_point("node_item_name_confirm")

        # 2、注册条件路由边
        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
            {
                "node_answer_output": "node_answer_output",
                "node_multi_search": "node_multi_search"
            }
        )

        # 3. 并发执行搜索
        self.workflow.add_edge("node_multi_search", "node_web_search_mcp")
        self.workflow.add_edge("node_multi_search", "node_search_embedding")
        self.workflow.add_edge("node_multi_search", "node_search_embedding_hyde")

        # 4. 多路搜索结果合并
        self.workflow.add_edge("node_search_embedding", "node_json")
        self.workflow.add_edge("node_search_embedding_hyde", "node_json")
        self.workflow.add_edge("node_web_search_mcp", "node_json")

        # 5. 合并 -> 排序 -> 重排 -> 生成 -> 结束
        self.workflow.add_edge("node_json", "node_rrf")
        self.workflow.add_edge("node_rrf", "node_rerank")
        self.workflow.add_edge("node_rerank", "node_answer_output")
        self.workflow.add_edge("node_answer_output", END)

    def compile(self):
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    def run(self, initial_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        """
           统一执行入口，支持切换invoke/stream
           :param initial_state:  初始状态对象
           :param stream: 是否是流式输出
           :return: 执行完成后的状态对象
        """
        if not self._compiled_app:
            self.compile()
        if stream:
            return self._compiled_app.stream(initial_state)
        else:
            return self._compiled_app.invoke(initial_state)

    @classmethod
    def create_and_run(cls, init_state: QueryGraphState, stream: bool = False):
        """
        快捷方法：创建工作流实例并且执行
        :param init_state:
        :param stream: 是否流式输出
        :return:
        """

        workflow = cls()

        return workflow.run(init_state, stream)



