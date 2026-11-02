# processor/query_process/nodes/node_item_name_confirm.py
"""
    这个节点主要干了 4 件事：

        1. 提取与改写 ：结合历史对话提取商品名，并将模糊问题改写为完整独立的精准问题。
        2. 向量化检索 ：将提取出的商品名在 Milvus 向量库中进行混合搜索。
        3. 标准化对齐 ：根据评分高低自动对齐标准型号，或生成反问让用户手动确认。
        4. 同步历史记录 ：将改写后的问题、确认的商品名和处理状态实时写入 MongoDB 数据库。
"""

import json
from typing import Dict, Tuple, List

from langchain_core.messages import SystemMessage, HumanMessage

from processor.config.config import lm_config, milvus_config
from processor.query_process.base import NodeBase
from processor.query_process import prompt
from processor.query_process.state import QueryGraphState
from processor.tool.logger import logger
from processor.utils.embedding_utils import generate_embeddings
from processor.utils.llm_utils import get_llm_client
from processor.utils.milvus_utils import create_hybrid_search_requests, hybrid_search
from processor.utils.mongo_history_utils import get_recent_messages, save_chat_message, format_json, \
    update_message_item_names


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # 步骤1：参数校验
        session_id, original_query = self._step1_validate_param(state)

        # 步骤2：获取历史会话记录
        history = get_recent_messages(session_id)
        # logger.info(f"步骤2：获取到 {len(history)} 条历史消息")
        # # 更新状态
        # state["history"] = history

        # 步骤3：用户初始消息保存
        message_id = save_chat_message(session_id, "user", original_query)

        # 步骤4：结合历史对话提取商品名，并改写用户问题（调用LLM）
        extract_res = self._step4_extract_info(original_query, history)
        item_names = extract_res.get("item_names")
        rewritten_query = extract_res.get("rewritten_query")

        # 5. & 6. 如果有提取到商品名，进行搜索和对齐
        align_result = {}
        if len(item_names) > 0:
            query_results = self._step5_vectorize_and_query(item_names)
            align_result = self._step6_align_item_names(query_results)
        else:
            logger.info("未提取到商品名，跳过向量检索")
        #
        # # 7. 检查确认状态
        dict_result = self._step7_check_confirmation(align_result, history)
        logger.info(f"步骤7：确认状态：{dict_result}")
        #
        # # 8. 写入最终历史
        self._step8_write_history(dict_result, session_id, original_query, rewritten_query, message_id)

        # # 9. 返回结果
        return {
            "history": get_recent_messages(session_id),
            "rewritten_query": rewritten_query,
            "item_names": dict_result.get("item_names"),
            "answer": dict_result.get("answer"),
        }


    def _step1_validate_param(self, state: QueryGraphState) -> Tuple:

        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("session_id 不能为空")

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("original_query 不能为空")

        return session_id, original_query

    def _step4_extract_info(self, original_query: str, history: list) -> Dict:

        try:
            # 1. 构造历史对话文本，拼接为"角色: 内容"的格式，供LLM做上下文理解
            history_text = ""

            # reversed: 将倒序变成正序
            for msg in history:
                history_text += f"{msg.get('role')}: {msg.get('text')}\n"

            # 2. 处理和动态拼接提示词
            # 为了把大括号当作 “普通字符” 保留下来，
            # 用双大括号 {{ 表示普通的左大括号 {，双大括号 }} 表示普通的右大括号 }。
            user_prompt = prompt.ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_text,
                original_query=original_query
            )

            # 3. 构建大模型对话消息
            messages = [
                SystemMessage(content=prompt.ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]

            # 4. 调用大模型并获取返回结果
            llm_client = get_llm_client(model=lm_config.item_model, json_mode=False)
            result = llm_client.invoke(input=messages)
            content = result.content

            logger.info(f"LLM返回结果：{content}")

            # 5. 数据清洗：处理LLM可能返回的代码块格式（如```json ... ```），去除包裹符
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")

            # 6. 数据解析：将JSON字符串转为字典
            result = json.loads(content)

            # 7. 健壮性处理：确保字段存在
            # 确保返回结果包含item_names字段，无则设为空列表
            if "item_names" not in result:
                result["item_names"] = []

            # 确保返回结果包含rewritten_query字段，无则复用原始查询
            if "rewritten_query" not in result:
                result["rewritten_query"] = original_query

            # 8. 给item_names 去除空格
            result["item_names"] = [item_name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
                                    for item_name in result["item_names"]]

            # 9、返回解析后的提取结果
            return result
        except Exception as e:
            # 捕获所有异常（如LLM调用失败、JSON解析失败等），记录错误日志
            logger.exception(f"大模型调用异常：{e}")
            # 异常时返回默认结果：空商品名列表+原始查询
            return {"item_names": [], "rewritten_query": original_query}

    def _step5_vectorize_and_query(self, item_names: list):

        """
        把分析出的item_names逐个向量化，并执行混合搜索，获取匹配评分
        :return: 格式：
           [
               {
                   "extracted_name": "hak180烫金机"
                   "matches": [                          # 该商品名的TopN匹配结果，无则空列表
                       {
                           "item_name": "BrotherHAK180烫金机D01WD7001-00",  # Milvus中存储的标准化商品名
                           "score": 0.80                  # 混合搜索的相似度评分（0-1，越高越相似）
                       },
                       {
                           "item_name": "HAK170烫金机",  # Milvus中存储的标准化商品名
                           "score": 0.80                  # 混合搜索的相似度评分（0-1，越高越相似）
                       },
                   ]
               },
               ...
           ]
       """
        # 1、初始化最终返回结果列表，存储每个商品名的向量化查询结果
        results = []

        # 2、获取集合名
        collection_name = milvus_config.item_name_collection

        # 3、批量生成双向量
        embeddings = generate_embeddings(item_names)
        dense_vectors = embeddings.get("dense")
        sparse_vectors = embeddings.get("sparse")

        # 4、遍历每个商品名称，逐个执行向量搜索
        for i, item_name in enumerate(item_names):
            # 构造Milvus混合搜索请求对象，传入稠/稀疏向量，指定返回Top5匹配结果
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vectors[i],
                sparse_vector=sparse_vectors[i],
                limit=5
            )

            # 执行混合搜索，获取数据库中的匹配结果和评分
            search_res = hybrid_search(
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                norm_score=True,  # 开启评分归一化，统一评分量级为0-1
                limit=5,
                output_fields=["item_name"],  # 指定返回Milvus中存储的商品名字段
            )
            hits = search_res[0] if search_res else []

            matches = [
                {
                    "item_name": hit.get("entity").get("item_name"),
                    "score": hit.get("distance"),
                }
                for hit in hits
            ]

            # 将当前商品名的原始名称+匹配结果，封装后加入最终结果列表
            results.append({
                "extracted_name": item_name,
                "matches": matches
            })
        # 返回所有商品名的向量化+搜索结果列表
        return results

    def _step6_align_item_names(self, query_results: List[Dict]) -> Dict:
        """
        6 根据Milvus搜索评分，逐个对齐step4提取的item_names，生成「确认商品名」和「候选商品名」
        对齐规则（优先级a>b>c）：
                a  如果只评分高于0.85 → 直接确认该商品名
                b  如果无0.85分以上结果 → 取分数≥0.6的最高前3个作为候选
                c  如果无0.6分及以上结果 → 不返回任何商品名（确认+候选均为空）
        :param query_results: 列表[字典] - step5的返回结果，每个商品名的搜索匹配数据（格式同step5返回值）
        :return: 字典 - 商品名对齐结果，包含确认列表和候选列表，格式：
                 {
                     "confirmed_item_names": ["确认商品名1", "确认商品名2"],  # 去重后的确认商品名，无则空列表
                     "options": ["候选商品名1", "候选商品名2", ...]          # 去重后的候选商品名，无则空列表
                 }
        """
        # 1、初始化确认商品名列表（符合高置信度规则的商品名）
        confirmed_item_names: List[str] = []

        # 2、初始化候选商品名列表（低置信度，需用户确认的商品名）
        options: List[str] = []

        logger.info(f"待处理的数据：{query_results}")
        for item in query_results:
            # 获取匹配的商品名
            matches = item.get("matches")
            # 若无匹配结果，直接跳过当前商品名的对齐
            if not matches:
                continue

            # 筛选高置信度匹配结果：评分>0.85
            high = [m for m in matches if m.get("score", 0) > 0.85]

            # 筛选中置信度匹配结果：评分≥0.6（仅高置信度为空时生效）
            mid = [m for m in matches if m.get("score", 0) >= 0.6]

            # a 所有评分高于0.85的都可以直接确认
            if high:
                confirmed_item_names.extend([m.get("item_name") for m in high])
                continue

            # 规则b: 无0.85分以上结果，取≥0.6分的最高前3个作为候选
            # 注：高置信度列表high为空时才会走到此处（规则a/b均不满足）
            if mid:
                options.extend([m.get("item_name") for m in mid[:3]])
                continue
            # 规则c: 无0.6分及以上结果 → 不做任何操作，确认+候选列表均为空
        # 返回最终对齐结果：确认列表和候选列表均做去重处理（list(set())）
        return {
            "confirmed_item_names": list(set(confirmed_item_names)),  # 去重，避免重复确认
            "options": list(set(options))  # 去重，避免重复候选
        }

    def _step7_check_confirmation(self, align_result, history):

        """
        Step7：对齐后的商品名状态，分3种分支更新state，并同步更新历史消息的商品名关联
        """
        # 从对齐结果中提取确认商品名列表，无则空列表
        confirmed_item_names = align_result.get("confirmed_item_names", [])

        # 从对齐结果中提取候选商品名列表，无则空列表
        options = align_result.get("options", [])

        # 分支A：有确认的商品名
        if confirmed_item_names:
            # 收集历史消息中未关联商品名的消息ID（需批量更新关联）
            ids_to_update = []
            for msg in history:
                if not msg.get("item_names"):  # 仅更新item_names为空的历史消息
                    mid = msg.get("_id")  # 提取消息唯一ID
                    if mid:
                        ids_to_update.append(str(mid))  # 转为字符串，避免ID格式

            if ids_to_update:
                update_message_item_names(ids_to_update, confirmed_item_names)

            return {
                "item_names": confirmed_item_names,
                "answer": ""
            }

        # 分支B：无确认商品名，但有候选商品名（中置信度，需用户明确）
        if options:
            # 候选商品名拼接为字符串，格式："商品1、商品2、商品3"
            options_str = "、".join(options)
            # 构造向用户确认的提示语
            answer = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
            logger.info(f"有候选商品名，返回给用户确认：{options}")
            # 更新会话状态
            return {
                "item_names": [],
                "answer": answer
            }
        return {
            "item_names": [],
            "answer": "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
        }

    def _step8_write_history(
            self,
            dict_result,
            session_id,
            original_query,
            rewritten_query,
            message_id
    ):
        """
            8 把本次处理的核心信息（用户问题、助手答案、商品名、改写查询）写入MongoDB的会话历史
            包含2个核心操作：1. 写入助手答案（若有）；2. 更新用户原始问题的关联信息
        """
        # 若会话状态中有助手答案（分支B/C），写入助手消息到历史
        if dict_result.get("answer"):
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=dict_result.get("answer")
            )
        # 强制更新本次用户原始问题的关联信息（核心：补充改写查询、商品名)
        else:
            save_chat_message(
                session_id=session_id,
                role="user",
                text=original_query,
                rewritten_query=rewritten_query,
                item_names=dict_result.get("item_names",[]),
                message_id=message_id
            )

