import os
from datetime import datetime
from typing import List, Dict

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

class MongoUtil:
    def __init__(self):
        # mongo连接对象 mongodb://192.168.200.128:27017
        self.client = MongoClient(os.getenv("MONGO_URL"))
        # 数据库
        self.db = self.client[os.getenv("MONGO_DB_NAME")]
        # 集合
        self.chat_message = self.db["chat_message"]

# 返回MongoUtil对象
def get_mongo_client()->MongoUtil:
    return MongoUtil()

# 1 获取用户前10条会话记录
# session_id
def get_recent_message(session_id:str,
                       limit:int=10)->List[Dict]:
    # 获取MongoUtil对象
    mongo_client = get_mongo_client()
    # 构建查询条件
    query = {"session_id": session_id}
    # 执行查询
    cursor = (mongo_client.chat_message.find(query)
                                       .sort("ts",-1)
                                       .limit(limit))
    message = list(cursor)
    return message

# 2 保存会话记录
# session_id:会话id
# role：角色 (user/assistant)
# text：会话内容
# rewritten_query：重写之后用户问题
# item_names：商品名称
# message_id：每个文档在MongoDB主键值
def save_chat_message(session_id: str,
                       role: str,
                       text: str,
                       rewritten_query: str = "",
                       item_names: List[str] = None ,
                       message_id:str=None) -> str:
    # 获取mongo_client对象
    mongo_client = get_mongo_client()

    # 获取当前时间戳
    ts = datetime.now().timestamp()

    # 构建添加数据
    data = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "item_names": item_names,
        "ts": ts
    }

    # 调用方法开始执行
    # message_id存在，更新；否则添加
    if message_id:
        # 更新
        mongo_client.chat_message.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": data},
        )
        return message_id
    else:
        result = mongo_client.chat_message.insert_one(data)
        return str(result.inserted_id)

# 3 清空会话记录
def clear_chat_message(session_id: str):
    mongo_client = get_mongo_client()
    result = mongo_client.chat_message.delete_many({"session_id": session_id})
    return str(result.deleted_count)


