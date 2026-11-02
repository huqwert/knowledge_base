# processor/query_process/nodes/node_rerank.py

"""
    这个节点是知识库检索的“精修师”，它对之前所有步骤召回的文档进行 二次精细排序 。
"""
from typing import List, Dict, Any

from processor.config.config import reranker_http_config
from processor.query_process.base import NodeBase
from processor.query_process.state import QueryGraphState
from processor.tool.logger import logger
from processor.utils.mongo_history_utils import format_json
from processor.utils.reranker_http_utils import rerank_documents


class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    # -----------------------------
    # Rerank / TopK 全局常量（不从 state 读取）
    # -----------------------------
    # 动态 TopK 硬上限：最多取前 N 条（<=10）
    RERANK_MAX_TOPK: int = 10
    # 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
    RERANK_MIN_TOPK: int = 3  # 总数最少条数

    # 断崖阈值（相对）
    RERANK_GAP_RATIO: float = 0.25
    # 断崖阈值（绝对）
    RERANK_GAP_ABS: float = 0.10

    def process(self, state: QueryGraphState):

        # 1. 合并多源文档
        merged_multi_docs: List[Dict[str, Any]] = self._step1_merge_multi_source_docs(state)

        # 2. Rerank 精排
        reranked_docs: List[Dict[str, Any]] = self._step2_rerank_merged_docs(state, merged_multi_docs)
        
        # 3. 动态 Top_K 截取
        cutoff_docs = self._step3_cliff_cutoff(reranked_docs)

        # # 4. 返回state
        return {
            "reranked_docs": cutoff_docs
        }

    def _step1_merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:

        final_docs = []
        for rrf_doc in state.get("rrf_chunks", []):
            format_rrf_doc = {
                "content": rrf_doc.get("content"),
                "title": rrf_doc.get("title"),
                "chunk_id": rrf_doc.get("chunk_id"),
                "url":None,
                "source": "local"
            }
            final_docs.append(format_rrf_doc)

        for web_doc in state.get("web_search_docs", []):
            format_web_doc = {
                "content": web_doc.get("snippet"),
                "title": web_doc.get("title"),
                "chunk_id": None,
                "url": web_doc.get("url"),
                "source": "web"
            }
            final_docs.append(format_web_doc)

        return final_docs

    def _step2_rerank_merged_docs(self, state: QueryGraphState, merged_multi_docs: List[Dict]) -> List[Dict]:
        """
        使用 Reranker 模型对文档进行精排
        """
        try:
            # 获取文档列表的conten字段组成列表
            query = state.get("rewritten_query", "")
            contents = [doc.get("content") for doc in merged_multi_docs]
            # 调用Rerank模型：交叉编码器（精排阶段）
            # Query 和 Document 联合编码，精度更高
            rerank_scores = rerank_documents(query, contents)

            scored_docs = [{"score":score,**doc} for score,doc in zip(rerank_scores,merged_multi_docs)]

            sorted_score_docs = sorted(scored_docs, key=lambda x: x.get("score"), reverse=True)

            return sorted_score_docs
        except Exception as e:
            raise ValueError(f"重排序过程中出现错误: {e}")

    def _step3_cliff_cutoff(self, ranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """断崖检测截断：相邻得分差距超过阈值时截断。"""
        if not ranked_docs:
            return []

        upper_bound = min(self.RERANK_MAX_TOPK,len(ranked_docs))
        lower_bound = min(self.RERANK_MIN_TOPK, upper_bound)

        # 默认值：取满硬上限（最多10条）
        cutoff_pos = upper_bound

        # 遍历范围：从min_topk-1到max_topk-2（索引从0开始），检测相邻两个文档的分数差
        # 例：min_topk=3，max_topk=10 → 遍历i=2,3,4,5,6,7,8（对应第3~9条文档，检测与下一条的差距）
        for idx in range(lower_bound - 1,upper_bound - 1):

            current_score = ranked_docs[idx].get("score")
            next_score = ranked_docs[idx + 1].get("score")

            # 触发断崖截断条件：绝对差距≥绝对阈值 OR 相对差距≥相对阈值
            # 满足任一条件，说明下一条文档相关性骤降，截断在当前位置
            if current_score is None or next_score is None:
                continue

            # 计算相邻文档的分数绝对差距（因已降序，gap≥0）
            abs_gap = current_score - next_score

            # 计算相对差距：绝对差距 / 当前文档分数（+1e-6避免除数为0/极小值，防止程序报错）
            # 1e-6 是 Python 中科学计数法的写法，等价于 0.000001（10 的负 6 次方，也就是百万分之一）。
            rel_gap = abs_gap / (abs(current_score) + 1e-6)

            if abs_gap >= self.RERANK_GAP_ABS or rel_gap >= self.RERANK_GAP_RATIO:
                cutoff_pos = idx + 1
                logger.info(f"断崖检测: 位置 {idx + 1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}")
                break
                
        return ranked_docs[:cutoff_pos]




