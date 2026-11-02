from typing import Optional

from langgraph.constants import END,START
from langgraph.graph import StateGraph

from processor.import_process.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_process.nodes.node_document_split import NodeDocumentSplit
from processor.import_process.nodes.node_entry import NodeEntry
from processor.import_process.nodes.node_import_milvus import NodeImportMilvus
from processor.import_process.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_process.nodes.node_md_img import NodeMDImg
from processor.import_process.nodes.node_pdf_to_md import NodePDFToMD
from processor.import_process.state import ImportGraphState
from processor.tool.logger import logger

class KBImportWorkflow:
    def __init__(self):
        # 1. 初始化工作流
        self.workflow = StateGraph(ImportGraphState)
        # 2. 初始化节点
        self._init_nodes()
        # 3. 注册节点到工作流
        self._register_nodes()
        # 4. 设置入口节点与路由
        self._setup_routes()
        # 5.编译工作流（懒加载/延迟加载，第一次执行的时候编译，第二次开始则无需编译）
        self._compile_app: Optional[object] = None

    def _init_nodes(self):
       self.node_entry = NodeEntry()
       self.node_pdf_to_md = NodePDFToMD()
       self.node_md_img = NodeMDImg()
       self.node_document_split = NodeDocumentSplit()
       self.node_item_name_recognition = NodeItemNameRecognition()
       self.node_bge_embedding = NodeBGEEmbedding()
       self.node_import_milvus = NodeImportMilvus()

    def _register_nodes(self):
        self.workflow.add_node("node_entry", self.node_entry)
        self.workflow.add_node("node_pdf_to_md", self.node_pdf_to_md)
        self.workflow.add_node("node_md_img", self.node_md_img)
        self.workflow.add_node("node_document_split", self.node_document_split)
        self.workflow.add_node("node_item_name_recognition", self.node_item_name_recognition)
        self.workflow.add_node("node_bge_embedding", self.node_bge_embedding)
        self.workflow.add_node("node_import_milvus", self.node_import_milvus)


    def _rout_after_entry(self,state: ImportGraphState):
        if state.get("is_md_read_enabled"):
            return "md_read"
        elif state.get("is_pdf_read_enabled"):
            return "pdf_read"
        return END

    def _setup_routes(self):
        # 设置入口节点
        self.workflow.set_entry_point("node_entry")
        # 注册条件边
        self.workflow.add_conditional_edges(
            "node_entry",
            self._rout_after_entry,
            {
                "md_read": "node_md_img",
                "pdf_read": "node_pdf_to_md",
                END: END
            }
        )
        # 注册静态顺序边
        self.workflow.add_edge(START, "node_pdf_to_md")
        self.workflow.add_edge("node_pdf_to_md", "node_md_img")
        self.workflow.add_edge("node_md_img", "node_document_split")
        self.workflow.add_edge("node_document_split", "node_item_name_recognition")
        self.workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
        self.workflow.add_edge("node_bge_embedding", "node_import_milvus")
        self.workflow.add_edge("node_import_milvus", END)

    def compile(self):

        if not self._compile_app:
            self._compile_app = self.workflow.compile()
        return self._compile_app

    def run(self,init_state: ImportGraphState,stream: bool = False):
        if not self._compile_app:
            self.compile()
        if stream:
            return self._compile_app.stream(init_state)
        else:
            return self._compile_app.invoke(init_state)

    @classmethod
    def create_and_run(cls, init_state: ImportGraphState, stream: bool = False):

        workflow = cls()
        return workflow.run(init_state, stream)
















