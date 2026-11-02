from collections import defaultdict
from typing import Dict, List

# ---------------------------
# 内存态任务追踪（单进程）
# ---------------------------
# key: task_id
# value: 节点名列表（原始英文/节点ID）

# defaultdict(list): 只要访问不存在的 key，自动帮你初始化 []
_tasks_running_list: Dict[str, List[str]] = defaultdict(list)
_tasks_done_list: Dict[str, List[str]] = defaultdict(list)
_tasks_duration: Dict[str, Dict[str, float]] = defaultdict(dict)

# key: task_id
# value: status 字符串（如 processing/completed/failed）
_tasks_status: Dict[str, str] = {}

# key: task_id
# value: 任务结果（例如 query 的 answer）

# 只要访问不存在的 key，自动帮你初始化 {}
_tasks_result: Dict[str, Dict[str, str]] = defaultdict(dict)


TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 节点名 -> 中文名映射（用于前端展示）
# 说明：这里的 key 应与 LangGraph 的 add_node("xxx", ...) 中的节点名一致。
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_item_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_milvus": "导入向量库",

    # --- Query 流程节点---
    "node_item_name_confirm": "确认问题产品",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "切片搜索(假设性文档)",
    "node_multi_search": "多路搜索",
    "node_join": "多路搜索合并",
}


# def _ensure_task(task_id: str) -> None:
#     """确保 task_id 对应的数据结构已初始化。"""
#     if task_id not in _tasks_running_list:
#         _tasks_running_list[task_id] = []
#     if task_id not in _tasks_done_list:
#         _tasks_done_list[task_id] = []
#     if task_id not in _tasks_result:
#         _tasks_result[task_id] = {}

def _to_cn(node_name: str) -> str:
    """将节点名转换为中文展示名；若无映射则返回原名。"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)

def add_running_task(task_id: str, node_name: str) -> None:
    """
    添加“正在运行”的节点任务。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)
    """
    # _ensure_task(task_id)

    # 1. 获取当前任务的运行节点列表（利用 defaultdict 自动初始化特性）
    running = _tasks_running_list[task_id]

    # 2. 将当前节点加入运行列表（并做去重判断，防止重复添加）
    if node_name not in running:
        running.append(node_name)

def add_done_task(task_id: str, node_name: str) -> None:
    """
    添加“已完成”的节点任务。
    注意：添加已完成任务时，会把同名的“正在运行”任务删除。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)
    """
    # _ensure_task(task_id)

    # 1) 从 running 中移除同名节点（可能出现重复，移除所有）
    # running = _tasks_running_list[task_id]
    # _tasks_running_list[task_id] = [n for n in running if n != node_name]

    # 1. 如果该节点还在运行列表中，则将其移出（表示该节点已结束运行）
    if node_name in _tasks_running_list[task_id]:
        _tasks_running_list[task_id].remove(node_name)

    # 2. 获取当前任务的已完成节点列表
    done = _tasks_done_list[task_id]

    # 3. 将当前节点加入已完成列表（做去重判断，防止重复标记）
    if node_name not in done:
        done.append(node_name)

    # TODO


def get_running_task_list(task_id: str) -> List[str]:
    """
    获取正在运行节点列表（中文展示）。
    """
    # _ensure_task(task_id)

    # 获取指定任务运行中的节点列表，并统一转换为中文名返回
    running = _tasks_running_list.get(task_id, [])
    return [ _to_cn(n)  for n in running]


def get_done_task_list(task_id: str) -> List[str]:
    """
    获取已完成节点列表（中文展示）。
    """
    # _ensure_task(task_id)

    # 获取指定任务已完成的节点列表，并统一转换为中文名返回
    done = _tasks_done_list.get(task_id, [])
    return [_to_cn(n) for n in done]


def get_task_status(task_id: str ) -> str:
    """
    获取当前任务状态。

    参数：
    - task_id: 任务ID

    返回：
    - str: 状态名称；如果未设置过则返回空字符串
    """

    # 安全获取指定任务的总体运行状态，若不存在则返回空字符串
    return _tasks_status.get(task_id, "")


def update_task_status(task_id: str, status_name: str) -> None:
    """
    更新任务状态。

    参数：
    - task_id: 任务ID
    - status_name: 状态名称（字符串）
    """

    # 更新指定任务的总体运行状态（如 processing 等）
    _tasks_status[task_id] = status_name


def set_task_result(task_id: str, key: str, value: str) -> None:
    """
    存储任务结果字段（如 answer / error）。
    """
    # _ensure_task(task_id)
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """
    获取任务结果字段（如 answer / error）。
    """
    # _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def add_node_duration(task_id: str, node_name: str, duration: float) -> None:
    """记录节点耗时（秒）"""
    cn_name = _to_cn(node_name)
    _tasks_duration[task_id][cn_name] = round(duration, 2)

def get_node_durations(task_id: str) -> Dict[str, float]:
    """获取所有节点的耗时"""
    return dict(_tasks_duration.get(task_id, {}))

def get_task_info(task_id: str) -> Dict[str, any]:
    """
    获取任务的全局信息（状态 + 运行中节点 + 已完成节点）
    :param task_id: 任务ID
    :return: 包含 status、running_list、done_list 的字典
    """
    return {
        "status": get_task_status(task_id),
        "running_list": get_running_task_list(task_id),
        "done_list": get_done_task_list(task_id),
        "durations": get_node_durations(task_id)
    }