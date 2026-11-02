from langchain_openai import ChatOpenAI

from processor.config.config import lm_config

# key: 模型名 + json_mode
# value: 模型对象
_llm_client_cache = {}

def get_llm_client(model: str | None = None ,json_mode: bool = False) -> ChatOpenAI:
    # 1. 获取模型名称，如果参数没有指定则使用默认模型
    model = model or lm_config.lm_model

    # 2. 定义缓存key
    key = (model, json_mode)

    # 3. 获取全局唯一的模型对象
    if key in _llm_client_cache:
        return _llm_client_cache[key]

    # 4. 关闭思考模式
    # 注意：不同的模型思考模式默认开启状态不一样，因此这里我们统一设置为非思考模式
    extra_body = {"enable_thinking": False}

    # 5. 配置响应数据类型是否是json
    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type":"json_object"}

    # 6. 创建模型对象
    llm = ChatOpenAI(
        model=model,
        base_url=lm_config.base_url,
        api_key=lm_config.api_key,
        temperature=lm_config.lm_temperature,
        extra_body=extra_body,
        model_kwargs=model_kwargs
    )

    # 7. 缓存模型对象
    _llm_client_cache[key] = llm

    return llm
