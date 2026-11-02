# processor/query_process/nodes/node_search_embedding.py
"""
    这个节点负责根据 改写后的用户问题 ，在 限定的商品范围内 ，利用 BGEM3 混合检索（稠密+稀疏） 技术，
    从 Milvus 向量数据库中召回 Top5 最相关的知识切片。
"""
from processor.config.config import milvus_config
from processor.query_process.base import NodeBase
from processor.query_process.state import QueryGraphState
from processor.tool.logger import logger
from processor.utils.embedding_utils import generate_embeddings
from processor.utils.milvus_utils import escape_milvus_string, create_hybrid_search_requests, hybrid_search
from processor.utils.mongo_history_utils import format_json


class NodeSearchEmbedding(NodeBase):

    # 覆盖基类的 name 属性，标识节点名称
    name = "node_search_embedding"

    def process(self,state: QueryGraphState):
        """
        节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
        """
        try:
            # 1. 参数校验
            rewritten_query, item_names = self._step1_validate_param(state)

            # 2、向量检索
            res = self._step2_search_embedding(
                rewritten_query=rewritten_query,
                item_names=item_names
            )

            # 3、结果封装
            return {"embedding_chunks": res}

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            return {"embedding_chunks": []}

    def _step1_validate_param(self, state):

        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise ValueError("改写后的用户问题不能为空")

        item_names = state.get("item_names")
        if not item_names:
            raise ValueError("商品名称不能为空")

        return rewritten_query, item_names

    def _step2_search_embedding(self, rewritten_query, item_names):

        try:
            # 1、生成向量 (Dense + Sparse)
            embeddings = generate_embeddings([rewritten_query])
            # logger.info(f"生成的向量: {embeddings}")
            dense_vector = embeddings.get("dense")[0]
            sparse_vector = embeddings.get("sparse")[0]
            logger.info(f"生成的向量: {dense_vector}, {sparse_vector}")

            # 2、处理 item_names 中的引号，防止注入或语法错误
            expr = None
            if item_names:
                escaped = ', '.join(f'"{escape_milvus_string(name)}"' for name in item_names)
                expr = f"item_name in [{escaped}]"
                logger.info(f"过滤条件: {expr}")
            else:
                logger.info("未指定商品名过滤，将全库检索")

            # 3、构造Milvus混合搜索请求对象
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                expr=expr,
                limit=10
            )
            # 4、执行混合向量检索
            logger.info("开始执行 Milvus 混合检索...")
            res = hybrid_search(
                collection_name=milvus_config.chunks_collection,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                norm_score=True,
                limit=10,
                output_fields=["chunk_id", "content", "item_name"],  # 指定返回字段
            )
            # 5、构造并返回结果：若检索结果非空，取res[0]，否则返回空列表
            return res[0] if res else []
        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            raise


