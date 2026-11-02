import json
import os
from os.path import splitext

from processor.import_process.base import NodeBase
from processor.import_process.state import ImportGraphState
from processor.tool.logger import logger


class NodeEntry(NodeBase):
    """
    入口节点：任务分发
    """

    name = "node_entry"

    def process(self, state: ImportGraphState):

        logger.info(f"【{self.name}】，程序的入口节点")

        # 1. 参数非空校验
        local_file_path = state.get('local_file_path')
        if not local_file_path:
            raise ValueError("请指定文件路径")

        # 2. 提取文件名称
        file_title = splitext(os.path.basename(local_file_path))[0]

        # 3. 文件类型检查
        # 是否为pdf
        if local_file_path.endswith('.pdf'):
            logger.info(f"文件类型检查：{local_file_path}")

            return {
                "is_pdf_read_enabled": True,
                "pdf_path": local_file_path,
                "file_title": file_title
            }

        # 是否为md
        if local_file_path.endswith('.md'):
            logger.info(f"MD文件类型检查：{local_file_path}")

            return {
                "is_md_read_enabled": True,
                "md_path": local_file_path,
                "file_title": file_title
            }
        # 其他文件
        else:
            last_index = local_file_path.rfind('.')
            current_type = local_file_path[last_index+1:]
            raise ValueError(f"不支持的文件类型：{current_type}")



