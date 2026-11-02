from openai import OpenAI
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from processor.config.config import embedding_config, web_embedding_config
from processor.tool.logger import logger

# 1. 定义模型单例对象，避免重复初始化
_bge_m3_ef = None


# 2. 获取模型单例对象
def get_bge_m3_ef():
    """
    获取BGE-M3-EF模型单例对象，自动加载环境变量配置
    """

    # 2.1 如果模型已经被实例化，则获取实例化对象
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    # 2.2 如果模型没有被提前下载，这里也会自动下载
    _bge_m3_ef =  BGEM3EmbeddingFunction(
        model_name=embedding_config.bge_m3_path,
        device=embedding_config.bge_device,
        fp16=embedding_config.bge_fp16,
    )
    return _bge_m3_ef

_bge_v4_ef = None
def get_bge_v4_rf():
    """
        获取_bge_v4_ef模型单例对象，自动加载环境变量配置
        """

    # 2.1 如果模型已经被实例化，则获取实例化对象
    global _bge_v4_ef
    if _bge_v4_ef is not None:
        return _bge_v4_ef

    # 2.2 远程调用api key
    _bge_v4_ef = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        # 各地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
        api_key=web_embedding_config.web_embedding_api_key,
        # 以下是北京地域base-url，如果使用新加坡地域的模型，需要将base_url替换为：
        # https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
        base_url=web_embedding_config.web_embedding_base_url,
    )
    return _bge_v4_ef

# 3. 生成向量嵌入
def generate_embeddings(texts, model_name: str = None):
    """
    为文本生成向量嵌入:包含dense和sparse向量
    """
    if model_name:
        model = get_bge_v4_rf()
        embeddings = model.embeddings.create(
            model=model_name,
            input=texts,
        )

        # return embeddings.data[0].embedding
        return {
            "dense": [emb.embedding for emb in embeddings.data],
        }
    else:
        model = get_bge_m3_ef()
        embeddings = model.encode_documents(texts)
    # embeddings 的数据结构：
    # 稠密向量：这是一个包含两个 array 的列表。每个 array 长度为 1024，数据类型是 float16（半精度）。
    # 稀疏向量：这是一个 Compressed Sparse Row sparse array（压缩稀疏行矩阵），
    #         形状是 (2, 250002)，250002 代表模型词汇表的大小（大约有 25 万个可能的词），2表示当前是2个词的向量
    # return embeddings

    return {
        "dense": [emb.tolist() for emb in embeddings["dense"]],
        # "sparse": [dict(zip(row.indices, row.data)) for row in embeddings["sparse"]]
        "sparse": [{int(k): float(v) for k,v in zip(row.indices, row.data)} for row in embeddings["sparse"]]
    }

