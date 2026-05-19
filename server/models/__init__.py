"""
数据库模型包
统一导出所有ORM模型和db实例
"""
from flask_sqlalchemy import SQLAlchemy

# 创建SQLAlchemy实例，在app.py中初始化
db = SQLAlchemy()

from .user import User
from .knowledge_base import KnowledgeBase
from .document import Document
from .chat_history import ChatHistory
