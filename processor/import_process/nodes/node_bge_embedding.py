import json
from typing import List, Dict

from processor.import_process.base import NodeBase
from processor.import_process.state import ImportGraphState
from processor.tool.logger import logger
from processor.utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(NodeBase):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):
        """
        LangGraph核心节点：BGE-M3文本向量化处理
        1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
        2. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
        3. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用
        """

        # 步骤1：输入数据校验
        chunks = self._step1_validate_input(state)

        # 步骤2：批量生成双向量，为切片绑定向量字段
        output_data = self._step2_generate_embeddings(chunks)

        # 步骤3：返回
        return {
            "chunks": output_data
        }

    def _step1_validate_input(self, state: ImportGraphState) -> List[Dict]:
        """
        Step 1：输入数据有效性校验
        """

        chunks = state.get("chunks")

        if not chunks:
            raise ValueError("chunks不能为空")

        if not isinstance(chunks, list):
            raise ValueError("chunks数据类型不正确")

        return chunks

    def _step2_generate_embeddings(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:

        """
        Step 2: 批量生成向量
        1. 分批处理：避免一次性处理过多数据导致显存溢出（OOM）。
        2. 文本构造：将 item_name 和 content 拼接，增强语义（商品名作为核心特征前置）。
        3. 向量生成：调用模型批量生成 Dense（稠密）和 Sparse（稀疏）向量。
        """

        # 定义批处理数量
        batch_size = 3
        # 初始化一个空列表
        output_data = []

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            # 批量构造文本
            texts = [f"{chunk['item_name']} \n{chunk['content']}" for chunk in batch_chunks]

            # 批量生成向量
            vectors = generate_embeddings(texts)
            dense_vectors = vectors.get("dense")
            sparse_vectors = vectors.get("sparse")

            for j, text in enumerate(texts):
                chunk = batch_chunks[j]
                chunk["dense_vector"] = dense_vectors[j]
                chunk["sparse_vector"] = sparse_vectors[j]
                output_data.append(chunk)

        return output_data


