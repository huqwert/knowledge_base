import asyncio
import json
import logging
import queue
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request

from processor.utils.task_utils import get_task_status, get_done_task_list, get_running_task_list


class SSEEvent:
    PROGRESS = "progress"  # 任务节点进度
    DELTA = "delta"  # LLM 流式输出增量
    FINAL = "final"  # 最终完整答案
    ERROR = "error"  # 错误信息


# 全局 SSE 任务队列存储
# Key: task_id, Value: queue.Queue
sse_queues: Dict[str, queue.Queue] = {}


def create_sse_queue(task_id: str):
    """
    创建并注册一个 sse 队列
    """
    sse_queues[task_id] = queue.Queue()

def remove_sse_queue(task_id: str):
    """
    移除 sse 队列
    """
    sse_queues.pop(task_id)

def push_sse_event(task_id: str, event: str, data: Dict[str, Any]):
    """
    通过 task_id 推送事件到 SSE 队列
    """
    # 1. 获取 SSE 队列
    stream_queue = sse_queues.get(task_id)

    # 2. 将事件推送到队列
    if stream_queue:
        stream_queue.put({"event": event, "data": data})

async def event_generator(task_id: str, request: Request) -> AsyncGenerator:
    """
    流式输出结果的消费者
    1. 从sse队列中获取结果
    2. 封装队列中的数据以及事件类型为sse协议的数据包格式
    3. 将封装好的数据包yield出去
    """

    # 1. 校验
    while task_id not in sse_queues:
        await asyncio.sleep(0.5)

    # 2. 根据任务id 获取任务队列对象
    sse_queue = sse_queues.get(task_id)

    loop = asyncio.get_event_loop()

    # 3. 让当前线程一直从队列中获取数据【如果队列一旦有数据，就直接获取，如果队列没有数据，等一会，再问一下】
    try:
        while True:

            # 3.1 判断前端sse连接是否关闭（主动探测）--->FastApI:可以感知到：request
            if await request.is_disconnected():
                return
            try:
                # 3.2 从队列中获取(阻塞队列---)为了让事件循环不阻塞，
                msg = await loop.run_in_executor(None, sse_queue.get, True, 1)
                # 3.3 获取事件类型
                event_type = msg.get('event')
                # 3.4 获取事件数据
                event_data = msg.get('data')
                # 3.5 打包返回
                payload = json.dumps(event_data, ensure_ascii=False)
                yield f"event: {event_type}\ndata: {payload}\n\n"
            except queue.Empty:
                logging.info(f"队列为空...请稍等")
                continue
    except  (ConnectionResetError, BrokenPipeError) as e:
        # 客户端中断 关闭了窗口或者浏览器
        return

    except asyncio.CancelledError:
        # 服务端中断 协程被取消 重新抛出，让外层知道它被成功取消()
        raise
    finally:
        remove_sse_queue(task_id)

def push_progress(task_id: str):
    push_sse_event(task_id, SSEEvent.PROGRESS, {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })