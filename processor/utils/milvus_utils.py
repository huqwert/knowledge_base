from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker, RRFRanker

from processor.config.config import lm_config, milvus_config
from processor.tool.logger import logger

# 1. 定义一个全局单例对象
_milvus_client = None


# 2. 获取全局单例对象
def get_milvus_client():
    global _milvus_client

    # 2.1 获取当前客户端对象
    if _milvus_client is not None:
        return _milvus_client

    # 2.2 创建客户端对象
    _milvus_client = MilvusClient(uri=milvus_config.milvus_url)

    # 2.3 返回客户端对象
    return _milvus_client


def escape_milvus_string(value: str) -> str:
    """
    Milvus数据库过滤表达式中字符串的安全转义函数（防止解析失败）
    作用： 转义特殊字符（反斜杠、双引号），避免Milvus解析filter时报错
    参数 value: 需要转义的原始字符串
    返回 str: 转义后的安全字符串
    """
    # 转义反斜杠（\ → \\） 双引号（" → \"） 单引号（' → \'）
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return value


def create_hybrid_search_requests(
        dense_vector,
        sparse_vector,
        dense_params=None,
        sparse_params=None,
        expr=None,
        limit=5
):
    """
    创建混合搜索请求

    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
    :param dense_vector: 文本生成的稠密向量
    :param sparse_vector: 文本生成的稀疏向量
    :param dense_params: 稠密向量搜索参数，默认使用余弦相似度
    :param sparse_params: 稀疏向量搜索参数，默认使用内积相似度
    :param expr: 搜索过滤表达式，用于精准筛选数据
    :param limit: 单向量搜索返回结果数量，默认5
    :return: 搜索请求列表，包含[dense_req, sparse_req]
    """
    # 稠密向量默认搜索参数
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # 稀疏向量默认搜索参数
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # 构建稠密向量搜索请求类
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        limit=limit,
        expr=expr,
    )

    # 构建稀疏向量搜索请求类
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        limit=limit,
        expr=expr,
    )

    # 返回搜索请求列表
    return [dense_req, sparse_req]

def hybrid_search(
        collection_name,
        reqs,
        ranker_weights=(0.5, 0.5),
        norm_score=False,
        limit=5,
        output_fields=None,
        search_params=None
):
    """
    执行混合搜索

    :param collection_name: 集合名称
    :param reqs: 搜索请求列表，固定为[dense_req, sparse_req]
    :param ranker_weights: 加权融合权重，默认(0.5,0.5)，依次对应稠密/稀疏向量
    :param norm_score: 是否归一化评分后再融合，避免评分量级差异导致权重失效
    :param limit: 混合搜索最终返回结果数量，默认5
    :param output_fields: 需要返回的字段列表，默认返回item_name
    :param search_params: 搜索参数，如ef/topk等，默认None
    :return: 混合搜索结果列表，搜索失败返回None
    """
    try:
        # 初始化加权排名器：按权重融合稠密/稀疏向量的搜索结果
        # norm_score=True：先将两个向量评分归一化到0~1区间，再加权计算，避免一个得分特别大、另一个特别小导致权重失效。
        # 版本：V2.5
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        client = get_milvus_client()
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
            ranker=rerank,
        )
        # 动态计算所有查询返回的结果总数
        logger.info(f"Milvus 混合搜索完成")
        return res
    except Exception as e:
        raise RuntimeError(f"执行Milvus混合搜索时发生错误: {e}")



def hybrid_search_rrf(client, collection_name, reqs, rrf_k=60, limit=5,
                  output_fields=None, search_params=None):
    """
    基于 RRF（倒数秩融合）的 Milvus 多路混合搜索
    无需分数归一化、无需手动调权重，天然适配稠密+稀疏等异构召回通路

    Args:
        client: Milvus 客户端连接实例
        collection_name: 目标集合名称
        reqs: 多路搜索请求列表（如 [稠密向量请求, 稀疏向量请求]）
        rrf_k: RRF 平滑系数，默认 60；越小头部结果权重越强，越大长尾互补性越强
        limit: 最终返回的结果条数
        output_fields: 返回的文档字段列表，默认返回 item_name
        search_params: 单路搜索的参数配置

    Returns:
        混合排序后的搜索结果，执行失败返回 None
    """
    try:
        # 初始化 RRF 排名器
        rerank = RRFRanker(k=rrf_k)

        # 默认返回字段兜底
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行混合搜索
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus RRF混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        logger.error(f"Milvus RRF混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None

