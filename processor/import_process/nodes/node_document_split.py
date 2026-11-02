import json
import re
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.import_process.base import NodeBase
from processor.import_process.state import ImportGraphState
from processor.tool.logger import logger


class NodeDocumentSplit(NodeBase):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    # --- 配置参数 ---
    # 限制单个 Chunk 的最大长度，超过此长度将触发二次切分
    DEFAULT_MAX_CONTENT_LENGTH = 500
    # 合并短 Chunk 的阈值，小于此长度的相邻 Chunk 会被尝试合并
    MIN_CONTENT_LENGTH = 100
    # 窗口overlap
    DEFAULT_WINDOW_OVERLAP = 50

    def process(self, state: ImportGraphState):
        # 步骤1：校验数据（防御性编程）
        content, file_title = self._step1_get_inputs(state)

        # 步骤2：按MD标题进行初次切分
        sections, title_count, lines_count = self._step2_split_by_titles(content, file_title)

        # 步骤3：无标题兜底处理
        # sections = self._step3_handle_no_title(content, sections, title_count, file_title)

        # 步骤4：对chunck进行精细化处理（长切短合）
        sections = self._step4_refine_chunks(sections)

        # 步骤5：输出一些统计信息
        self._step5_print_stats(lines_count, sections)

        # 步骤6：chunck结果在磁盘上进行备份
        self._step6_backup(state, sections)

        # 步骤7：返回结果

        return {
            "chunks": sections
        }

    def _step1_get_inputs(self, state: ImportGraphState) -> Tuple[str,str]:
        """
        Step1：获取并预处理输入数据
        """
        # 非空校验
        file_title = state.get("file_title")
        if not file_title:
            raise ValueError("文本标题不能为空")

        md_content = state.get("md_content")
        if not md_content:
            raise ValueError("文本内容不能为空")

        return md_content, file_title

    def _step2_split_by_titles(self, content: str, file_title: str):
        """
        按照md标题进行初切
        :param content:
        :param file_title:
        :return:
        """

        # 1、定义标题正则
        # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
        # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
        # #{1,6}：匹配1-6个#（对应MD1-6级标题）
        # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
        # .+：标题文字至少1个字符（避免空标题）

        title_pattern = r'^\s*#{1,6}\s+.+'

        # 2. 输出需要的数据
        in_code_block = False  # 代码围栏：False，当前没有在代码围栏中，True：在
        current_lines = []  # 当前标题和下一个标题之间的文本内容
        sections = []  # 章节列表
        current_title = ""  # 当前章节的标题
        title_count = 0  # 标题数量

        # 3. 定义内部函数组装章节列表中的一个成员
        def _flush_section():

            if not current_lines:
                return
            nonlocal title_count
            title_count +=1
            sections.append({
                "file_title": file_title,
                "title": current_title or "无标题",
                "content": '\n'.join(current_lines)
            })

        # 数据清洗：逐行遍历。识别标题
        lines = content.split("\n")
        for line in lines:
            stripped_line = line.strip()

            # 识别代码围栏
            # if stripped_line.startswith("```"):
            #     in_code_block = not in_code_block

            # 识别代码围栏(加强版)
            code_pattern = r"^(`{3,}|~{3,})"
            code_match = re.match(code_pattern,stripped_line)
            if code_match:
                # 取出匹配到的代码围栏
                marker = code_match.group(1)

                if not in_code_block:
                    # 代码块开始
                    in_code_block = True
                    code_fence = marker
                    logger.info(f"识别带代码围栏开始：{marker}")
                else:
                    if code_fence == marker:
                        # 匹配到结束代码围栏
                        in_code_block = False
                        code_fence = None
                        logger.info(f"识别带代码围栏结束：{marker}")


            # 识别文档标题
            is_valid_title = not in_code_block and re.match(title_pattern, stripped_line)
            if is_valid_title:
                # 遇到标题则将上一个片段写入sections，再初始化新的章节
                _flush_section()
                current_title = stripped_line
                current_lines = [current_title]
                # title_count += 1
                logger.info(f"识别标题：{current_title}")

            else:
                # 否则，将行加入当前片段
                current_lines.append(line)
        # 将最后剩余的行放入section
        _flush_section()
        logger.info(f"文档粗切完成，识别标题数量：{title_count}，共{len(sections)}个章节，文档共{len(lines)}行")
        logger.info(sections)
        return sections, title_count, len(lines)

    def _step3_handle_no_title(
            self,
            content: str,
            sections: List[Dict[str, str]],
            title_count: int,
            file_title: str
    ) -> List[Dict[str, str]]:

        if title_count == 0:
            sections.append({
                "file_title": file_title,
                "title": "无标题",
                "content": content
            })
        return sections

    def _step4_refine_chunks(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:

        # 1. 切分超长章节
        refined_split = []
        for sec in sections:
            refined_split.extend(self._split_long_section(sec))
        logger.info(f"切分超长章节完成，共{len(refined_split)}个chunk")

        # 2. 合并过短章节
        final_sections = self._merge_short_sections(refined_split)
        logger.info(f"合并短章节完成，共{len(final_sections)}个chunk")

        # 3. 整合
        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title")
                sec["part"] = 0

        logger.info(f"最后整理完成，共{len(final_sections)}个chunk")

        return final_sections

    def _split_long_section(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        """
        长内容二次切分
        """
        content = section.get("content")
        # 1. 判断内容的长度，如果没有超过最大长度阈值，则无需切分
        if len(content) <= self.DEFAULT_MAX_CONTENT_LENGTH:
            return [section]

        # 2. 当文档中有特殊内容时（例如HTML的table）
        if '<table' in content:
            return [section]

        # 3. 定义标题前缀
        title = section.get("title")
        prefix = f"{title}\n\n "

        # 4. 计算切分后正文允许的可用长度： 总长度 - 标题的长度
        available_len = self.DEFAULT_MAX_CONTENT_LENGTH - len(prefix)
        if available_len <= 100:
            logger.warning(f"章节标题过长，无法进行二次切分：{title}")
            return [section]

        logger.info(f"开始对超长章节进行二次切分：{title}")

        # 5. 去掉content部分的title
        body = content
        if title and body.startswith(title):
            body = body[len(title):]

        # 6. 针对body进行递归切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,
            chunk_overlap=self.DEFAULT_WINDOW_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "]
        )

        # 定义子章节
        sub_sections = []

        # 递归切分
        for idx,chunk in enumerate(splitter.split_text(body), start=1):
            text = chunk.strip()
            if not text:
                continue

            # 组装chunk的完整内容
            full_text = prefix + text

            # 子章节信息封装
            sub_sections.append({
                "title": f"{title} - {idx}",
                "content": full_text,
                "parent_title": title,
                "part": idx,
                "file_title": section.get("file_title")
            })
        logger.info(f"章节{title}【二次切分】完成，共{len(sub_sections)}个子章节")
        return sub_sections

    def _merge_short_sections(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        短内容重新合并
        """

        # 1. 健壮性判断
        if not sections:
            logger.info("待合并列表为空")
            return []

        # 2. 定义最终合并结果列表
        merged_sections = []

        # 3. 定义当前section的临时变量
        current_chunk = None

        # 4. 遍历所有的section
        for sec in sections:

            # 4.1 第一个chunk直接作为待合并单元
            if current_chunk is None:
                current_chunk = sec
                continue

            # 4.2 当前单元的content是否不满足阈值设置的最小长度
            is_current_short = len(current_chunk["content"]) < self.MIN_CONTENT_LENGTH

            # 4.3 与下一个单元是否是同一个parent_title
            is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

            # 4.4 判断是否满足合并条件
            if is_current_short and is_same_parent:
                # 获取前缀标题
                parent_title = sec.get("parent_title","")
                # 获取本轮遍历的content
                next_content = sec.get("content")

                # 去掉content部分的前缀标题
                if parent_title and next_content.startswith(parent_title):
                    next_content = next_content[len(parent_title):]

                # 合并内容
                current_chunk["content"] += "\n\n" + next_content

                # 更新子Chunk序号：保留最新序号，便于溯源
                if "part" in sec:
                    current_chunk["part"] = sec["part"]

                logger.info(f"合并短Chunk：{current_chunk.get('parent_title')} → 累计长度{len(current_chunk['content'])}")

            else:
                # 如果content的长度超过最小阈值，则将当前单元合并到结果列表中
                merged_sections.append(current_chunk)
                current_chunk = sec
        # 将最后一个单元合并到结果列表中
        if current_chunk is not None:
            merged_sections.append(current_chunk)

        return merged_sections

    def _step5_print_stats(self, lines_count: int, sections: List[Dict[str, str]]) -> None:
        """
        Step5：输出文档切分统计信息（日志记录，便于监控/调试）
        :param lines_count: MD原始文本总行数
        :param sections: 最终处理后的Chunk列表
        """
        chunk_num = len(sections)
        # 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
        logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        logger.info(f"MD原始文本总行数：{lines_count}")
        logger.info(f"最终生成Chunk数量：{chunk_num}")

    def _step6_backup(self, state: ImportGraphState, sections: List[Dict[str, str]]) -> None:
        """
        Step6: Chunk结果本地JSON备份（便于调试/问题排查，保留处理结果）
        :param state: 项目状态字典，需包含md_dir（备份目录）
        :param sections: 最终处理后的Chunk列表
        """

        try:
            # 拼接备份文件路径：固定文件名，便于查找
            backup_path = Path(state.get("local_dir")) / state.get("file_title") / "chunks.json"
            # 写入JSON文件：保留中文/格式化缩进，便于人工查看
            with open(backup_path, "w", encoding="utf-8") as f:
                """
                sections是Python 嵌套数据结构（List[Dict[str, str]]，列表里装字典，字典里可能嵌套字符串 / 数字等），而普通文件写入
                （如f.write(sections)）仅支持写入字符串，直接写 Python 数据结构会报错。
                json.dump的核心作用就是：将 Python 原生数据结构（列表、字典、字符串、数字等）直接序列化并写入 JSON 文件，无需手动转换为字符串，
                同时保证数据格式规范、可跨语言 / 跨场景读取，完美适配「Chunk 列表备份」的需求。
                """
                json.dump(
                    sections,
                    f,
                    # 开启 True："title": "\u4e00\u7ea7\u6807\u9898"（乱码，无法直接看）；
                    # 开启 False："title": "一级标题"（正常中文，人工可直接阅读）。
                    ensure_ascii=False,  # 保留中文，不转义为\u编码
                    indent=2  # 格式化缩进，便于阅读
                )
            logger.info(f"Chunk结果备份成功，备份文件路径：{backup_path}")
        except Exception as e:
            # 备份失败仅记录日志，不终止主流程
            logger.error(f"Chunk结果备份失败，错误信息：{str(e)}", exc_info=False)



