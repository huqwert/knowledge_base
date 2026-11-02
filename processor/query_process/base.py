from abc import ABC, abstractmethod

from processor.query_process.state import QueryGraphState
from processor.tool.logger import logger
from processor.utils.sse_utils_sync import push_sse_event, SSEEvent
from processor.utils.task_utils import add_running_task, add_done_task, get_task_status, get_done_task_list, \
    get_running_task_list


class NodeBase(ABC):

    name: str = "node_base"

    def __init__(self):
        """
        强制子类设置name
        """
        if self.name == "node_base":
            raise ValueError(f"{self.__class__.__name__} 必须设置 name 属性")

    def __call__(self, state: QueryGraphState):

        try:


            task_id = state.get("task_id", "")
            # 1. 记录节点的运行状态:开始
            logger.info(f"{self.name} 开始执行...")
            add_running_task(task_id, self.name)
            # 2. 加入事件队列
            self._push_progress(task_id)

            # 3. 执行节点
            result = self.process(state)

            # 4. 记录节点的运行状态:结束
            add_done_task(task_id, self.name)

            # 5. 加入事件队列
            self._push_progress(task_id)
            logger.info(f"{self.name} 结束执行...")


            return result

        except Exception as e:
            logger.error(f"{self.name} 执行失败：{e}")
            raise

    @abstractmethod
    def process(self,state: QueryGraphState):
        """
        节点的核心处理逻辑
        :return:
        """
        pass

    def _push_progress(self, task_id: str):
        """
        在异步队列中放入事件
        """
        push_sse_event(task_id, SSEEvent.PROGRESS, {
            "status": get_task_status(task_id),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        })


