from http import HTTPStatus

import dashscope


from processor.config.config import lm_config, reranker_http_config
from processor.tool.logger import logger


def rerank_documents(query: str, documents: list[str]) -> list[float]:


    dashscope.api_key = reranker_http_config.api_key
    response = dashscope.TextReRank.call(
        model=reranker_http_config.model,
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        instruct=reranker_http_config.instruct

    )

    if response.status_code != HTTPStatus.OK:
        message = response.get("message")
        raise RuntimeError(f"DashScope rerank 调用失败: {message}")

    # logger.info(f"DashScope rerank 调用成功: {type(response)}")

    results = response.output.results

    # 保持索引对齐
    scores = [0.0] * len(documents)
    logger.info(f"DashScope rerank 调用成功: {results}")
    for item in results:
        index = item.get("index")
        score = item.get("relevance_score")
        scores[index] = score

    return scores

