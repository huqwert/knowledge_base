# processor/query_process/nodes/node_search_embedding_hyde.py
"""
    这个节点实现了 HyDE (Hypothetical Document Embeddings) 策略，核心逻辑是 先让 LLM 虚构一个“理想答案”，再用这个答案去向量库检索真实的文档 。
    它通过“LLM 生成假设性答案”来增强原始问题的语义信息，再进行混合向量检索，从而大幅提升对“语义匹配但字面不匹配”问题的召回能力。
"""
from processor.config.config import milvus_config
from processor.query_process.base import NodeBase
from processor.query_process.prompt import HYDE_PROMPT
from processor.query_process.state import QueryGraphState
from processor.tool.logger import logger
from processor.utils.embedding_utils import generate_embeddings
from processor.utils.llm_utils import get_llm_client
from processor.utils.milvus_utils import create_hybrid_search_requests, hybrid_search, escape_milvus_string
from processor.utils.mongo_history_utils import format_json


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) :
        try:
            # 1. 参数校验
            rewritten_query, item_names = self._step1_validate_param(state)

            # 2、生成假设性文档
            hyde_doc = self._step2_create_hyde_doc(rewritten_query)

            # 3、用“重写问题 + 假设文档”检索切片
            res = self._step3_search_embedding_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names
            )

            # 4、结果封装
            return {"hyde_embedding_chunks": res}

        except Exception as e:
            logger.exception(f"假设性文档向量搜索失败: {e}")
            return {"hyde_embedding_chunks": []}

    def _step1_validate_param(self, state):

        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise ValueError("问题参数错误")

        item_names = state.get("item_names")
        if not item_names:
            raise ValueError("商品名参数错误")

        return rewritten_query, item_names

    def _step2_create_hyde_doc(self, rewritten_query):

        try:
            logger.info("开始生成假设性文档")

            llm = get_llm_client()
            hyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)
            hyde_doc = llm.invoke(hyde_prompt).content
            return hyde_doc
        except Exception as e:
            logger.exception(f"模型调用失败: {e}")
            raise


    def _step3_search_embedding_hyde(self, rewritten_query, hyde_doc, item_names):

        try:
            # 1、生成向量 (Dense + Sparse)
            logger.info("开始用“重写问题 + 假设文档”检索切片")
            embeddings = generate_embeddings([rewritten_query + "\t" + hyde_doc])
            dense_vector = embeddings.get("dense")[0]
            sparse_vector = embeddings.get("sparse")[0]

            # 2、处理 item_names 中的引号，防止注入或语法错误
            expr = None
            if item_names:
                escaped = ', '.join(f'"{escape_milvus_string(name)}"' for name in item_names)
                expr = f"item_name in [{escaped}]"
                logger.info(f"过滤条件: {expr}")
            else:
                logger.info("未指定商品名过滤，将全库检索")

            reqs = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                expr=expr,
                limit=5
            )

            res = hybrid_search(
                collection_name=milvus_config.chunks_collection,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                norm_score=True,
                limit=10,
                output_fields=["chunk_id", "content", "item_name"], # 指定返回字段
            )
            # 5、构造并返回结果：若检索结果非空，取res[0]，否则返回空列表
            return res[0] if res else []
        except Exception as e:
            logger.exception(f"生成向量或创建搜索请求失败: {e}")
            raise



