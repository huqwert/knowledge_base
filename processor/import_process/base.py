import time
from abc import ABC, abstractmethod


from processor.import_process.state import ImportGraphState
from processor.tool.logger import logger
from processor.utils.task_utils import add_running_task, add_done_task, add_node_duration


class NodeBase(ABC):

    name: str = "node_base"

    def __init__(self):
        """
        强制子类设置name
        """
        if self.name == "node_base":
            raise ValueError(f"{self.__class__.__name__} 必须设置 name 属性")

    def __call__(self, state: ImportGraphState):

        try:

            logger.info(f"{self.name} 开始执行...")
            task_id = state.get("task_id", "")

            # 1. 开始：记录节点运行状态
            add_running_task(state["task_id"], self.name)


            # 2. 记录开始时间
            start_time = time.time()

            # 3. 执行节点
            result = self.process(state)

            # 4. 记录结束时间
            end_time = time.time()

            # 5. 结束：记录节点完成状态
            add_done_task(state["task_id"], self.name)

            # 6. 记录节点运行时间
            add_node_duration(task_id, self.name, end_time - start_time)

            logger.info(f"{self.name} 结束执行...")

            return result

        except Exception as e:
            logger.error(f"{self.name} 执行失败：{e}")
            raise

    @abstractmethod
    def process(self: ImportGraphState):
        """
        节点的核心处理逻辑
        :return:
        """
        pass


