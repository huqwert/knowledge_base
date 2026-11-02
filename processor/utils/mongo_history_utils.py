# processor/utils/mongo_history_utils.py
import json
from datetime import datetime
from typing import List, Any, Dict

from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING

from processor.config.config import mongo_config
from processor.tool.logger import logger


class HistoryMongoTool:

    def __init__(self):
        try:
            # 从环境变量中读取基本配置
            # 数据库连接
            self.mongo_url = mongo_config.mongo_url
            # 数据库名称
            self.db_name = mongo_config.mongo_db_name
            # 远程连接对象
            self.client = MongoClient(self.mongo_url)
            # 数据库对象
            self.db = self.client[self.db_name]
            # 集合对象
            self.chat_message = self.db["chat_message"]

            # 创建chat_message的索引结构：基于session_id正序，且基于ts倒序
            # create_index 自带幂等性
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            logger.info("MongoDB连接成功")
        except Exception as e:
            logger.error(f"MongoDB连接失败: {e}")
            raise


# 延迟加载
# _history_mongo_tool = None
# 预加载
_history_mongo_tool = HistoryMongoTool()
def get_history_mongo_tool():

    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()

    return _history_mongo_tool

def clear_history(session_id: str) -> int:
    """
    清空会话历史记录
    :param session_id:
    :return: 删除的文档的数量
    """
    try:
        mongo_tool = get_history_mongo_tool()
        result = mongo_tool.chat_message.delete_many({"session_id":session_id})
        logger.info("清空会话历史记录成功")
        return result.deleted_count
    except Exception:
        logger.error(f"清空会话历史记录失败:{session_id}")
        return 0


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        image_urls: List[str] = None,
        message_id: str = None
) -> str:
    """
    新增或更新会话消息
        :param session_id:
        :param role:
        :param text:
        :param rewritten_query:
        :param item_names:
        :param image_urls:
        :param message_id:
        :return: 当前消息的message_id
    """

    try:

        mongo_tool = get_history_mongo_tool()
        if message_id:
            document = {
                "session_id": session_id,
                "role": role,
                "text": text,
                "rewritten_query": rewritten_query,
                "item_names": item_names,
                "image_urls": image_urls,
            }
            mongo_tool.chat_message.update_one(
                {"_id": ObjectId(message_id)},
                {"$set": document}
            )
            return message_id
        else:
            document = {
                "session_id": session_id,
                "role": role,
                "text": text,
                "rewritten_query": rewritten_query,
                "item_names": item_names,
                "image_urls": image_urls,
                "ts": datetime.now().timestamp()
            }
            result = mongo_tool.chat_message.insert_one(document)
            return str(result.inserted_id)
    except Exception:
        logger.error(f"保存或更新失败：{session_id}")
        return 0

def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """
    更新当前聊天记录中的商品名字段（通过意图判断获取的）
    :param ids: 要更新的记录的message_id列表
    :param item_names: 新的商品名称
    :return: 更新的文档的数量
    """

    try:
        mongo_tool = get_history_mongo_tool()

        object_ids = [ ObjectId(id) for id in ids ]
        result = mongo_tool.chat_message.update_many(
            {"_id": {"$in": object_ids}},
            {"$set": {"item_names": item_names}}
        )
        return result.modified_count
    except Exception :
        logger.error("更新主体名称失败")
        return 0


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    获取聊天历史记录
    :param session_id:
    :param limit:
    :return:
    """

    try:
        mongo_tool = get_history_mongo_tool()

        cursor = mongo_tool.chat_message.find({"session_id":session_id}).sort("ts",DESCENDING).limit(limit)
        return list(cursor)
    except Exception:
        logger.error("获取最近历史消息失败")
        return []

# 定义自定义 JSON Encoder，解决原生json工具无法序列化ObjectId的问题
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def format_json(data: Any, indent: int = 4, ensure_ascii: bool = False) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, cls=MongoJSONEncoder)


