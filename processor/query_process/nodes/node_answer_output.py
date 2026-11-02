# processor/query_process/nodes/node_answer_output.py
"""
    这个节点是知识库查询的“最后一公里”，负责生成最终回答并交付给用户 。
"""
import re
from typing import List, Dict, Tuple

from processor.query_process.base import NodeBase
from processor.query_process.prompt import ANSWER_PROMPT
from processor.query_process.state import QueryGraphState
from processor.tool.logger import logger
from processor.utils.llm_utils import get_llm_client
from processor.utils.mongo_history_utils import save_chat_message
from processor.utils.sse_utils_sync import push_sse_event, SSEEvent


class NodeAnswerOutput(NodeBase):
    """
    节点功能: 答案生成
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_answer_output"

    MAX_CONTEXT_CHARS = 12000

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 1. 获取task_id
        task_id= state.get("task_id")

        # 2. 获取answer
        answer = state.get("answer")

        # 3. 判断是否有answer
        if answer:

            # 将事件结果放入异步队列
             push_sse_event(task_id, SSEEvent.FINAL, {"answer": answer})

        else:

            # 如果没有获取answer（已经进行了并发检索）：调用LLM生成答案，再将内容推给前端
            prompt = self._step1_construct_prompt(state)
            state["prompt"] = prompt

            # 3. 调用大模型输出答案
            answer = self._step2_generate_response(state, prompt)
            state["answer"] = answer

            # 3. 提取图片
            image_urls = self._step3_extract_images_from_docs(state.get("reranked_docs"))

            # # 4. 把答案写入到mongodb的history中
            if state.get("answer"):
                self._step4_write_history(state, image_urls=image_urls)

            # 5. 流式输出结束，发送 final 事件
            push_sse_event(task_id, SSEEvent.FINAL, {"image_urls": image_urls})

        return state

    def _step1_construct_prompt(self, state: QueryGraphState) -> str:

        """
        Step1：构建 Prompt
        根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组装 LLM 提示词
        """
        char_budget = self.MAX_CONTEXT_CHARS

        # 1. 获取问题和商品名
        # 优先使用重写后的问题
        question = state.get("rewritten_query")
        item_names = state["item_names"]

        # 2. 格式化上下文文档
        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )

        # 3. 格式化历史对话
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )

        # 4. 格式化 Item Names (提问商品)
        item_names_str = ", ".join(item_names) if item_names else "无指定商品"

        # 5. 组装提示词
        prompt = ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=item_names_str,
            question=question,
        )
        return prompt

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化重排序文档，带字符预算控制"""
        formatted_lines = []
        used_chars = 0

        # 从重排内容中，提取为资料字符串，不可超过限额
        # 优先使用结构化 reranked_docs（包含 source/chunk_id/url/score），便于约束与引用
        # ---------------------------------------------------------
        # 逻辑解释：
        # 1. 遍历重排序后的文档列表 (reranked_docs)，这些文档已经按相关性从高到低排序。
        # 2. 对每个文档提取关键信息 (text, source, chunk_id, url, title, score)。
        # 3. 构造 "元数据头 + 正文" 格式的字符串，例如：
        #    "[1] [local] [chunk_id=123] [score=0.95] [title=操作手册]
        #     这里是文档的正文内容..."
        # 4. 累加字符长度，如果超过 MAX_CONTEXT_CHARS (如 12000 字符)，则停止添加，
        #    确保 Prompt 长度在 LLM 的处理范围内，避免 Token 溢出。
        # ---------------------------------------------------------
        for idx, doc in enumerate(reranked_docs, start=1):
            content = doc.get("content")
            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field)).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化历史对话"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"
            used_chars += len(formatted_line) + 1

            if used_chars > char_budget:
                return "\n".join(formatted_lines), char_budget - used_chars

            formatted_lines.append(formatted_line)

        return "\n".join(formatted_lines), char_budget - used_chars

    def _step2_generate_response(self, state: QueryGraphState, prompt: str) -> QueryGraphState:
        """
        Step2：生成回答
        """
        # 获取 LLM 客户端
        llm = get_llm_client()

        # 获取task_id
        task_id = state.get("task_id")

        final_text = ""
        try:
            # 使用 stream 方法进行流式生成
            for chunk in llm.stream(prompt):
                delta = chunk.content
                if delta:
                    # 将增量内容放入队列
                    push_sse_event(task_id, SSEEvent.DELTA, {"delta": delta})
                    final_text += delta

        except Exception as e:
            push_sse_event(task_id, SSEEvent.ERROR, {"error": str(e)})
            logger.error(f"流式生成出错: {e}", exc_info=True)

        return final_text

    def _step3_extract_images_from_docs(self, docs):
        """
        Step3:从文档列表中提取图片URL
        """
        images = []
        seen = set()  # 用于去重，避免同一张图片重复出现
        if not docs:
            return []

        md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

        for i, doc in enumerate(docs):
            # 检查 text 字段中的 Markdown 图片 (主要针对 Local Chunk)
            text = doc.get("content")
            matches = md_img_pattern.findall(text)
            for img_url in matches:
                img_url = img_url.strip()
                if img_url and img_url not in seen:
                    seen.add(img_url)

        images = list(seen)
        return images

    def _step4_write_history(self, state: QueryGraphState, image_urls=None) -> QueryGraphState:
        """
        Step5：把本轮答案写入 MongoDB history
        """
        session_id = state.get("session_id")
        answer = state.get("answer")
        item_names = state.get("item_names")

        try:
            if answer:
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    rewritten_query="",
                    item_names=item_names,
                    image_urls=image_urls,
                    message_id=None
                )
        except Exception as e:
            # 写历史失败不应影响主链路
            logger.error(f"写入Mongo历史记录失败: {e}")

        return state