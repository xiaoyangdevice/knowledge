"""
项目配置文件
包含数据库、阿里云百炼API、Chroma等配置信息
"""
import os


class Config:
    """基础配置类"""

    # Flask密钥，用于JWT签名
    SECRET_KEY = os.environ.get('SECRET_KEY', 'enterprise-qa-secret-key-2024')

    # MySQL数据库配置（端口3308，密码123456）
    MYSQL_HOST = os.environ.get('MYSQL_HOST', '127.0.0.1')
    MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 3306))
    MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '110216')
    MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE', 'db_qa')

    # SQLAlchemy数据库连接URI
    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # JWT Token有效期（秒），默认24小时
    JWT_EXPIRATION = 86400

    # ========================================
    # 阿里云百炼平台API配置
    # ========================================
    
    # --- LLM对话模型配置（使用CodePlan API）---
    # CodePlan API密钥
    DASHSCOPE_LLM_API_KEY = os.environ.get('DASHSCOPE_LLM_API_KEY', 'sk-sp-df253f04599e496bbe534b2784b0----')
    # CodePlan API地址
    DASHSCOPE_LLM_BASE_URL = os.environ.get('DASHSCOPE_LLM_BASE_URL', 'https://coding.dashscope.aliyuncs.com/v1')
    # 大语言模型名称
    DASHSCOPE_LLM_MODEL = os.environ.get('DASHSCOPE_LLM_MODEL', 'qwen3-max-2026-01-23')
    
    # --- Embedding向量化模型配置（使用标准DashScope API）---
    # DashScope API密钥
    DASHSCOPE_EMBED_API_KEY = os.environ.get('DASHSCOPE_EMBED_API_KEY', 'sk-4e6e4e701016491c9186973102a1----')
    # DashScope API地址
    DASHSCOPE_EMBED_BASE_URL = os.environ.get('DASHSCOPE_EMBED_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    # Embedding模型名称
    # text-embedding-v2: 固定1536维，不支持dimension参数
    # text-embedding-v3: 支持1024/768/512维，可通过dimension参数设置
    DASHSCOPE_EMBED_MODEL = os.environ.get('DASHSCOPE_EMBED_MODEL', 'text-embedding-v2')
    # Embedding向量维度（仅text-embedding-v3支持此参数）
    DASHSCOPE_EMBED_DIMENSION = int(os.environ.get('DASHSCOPE_EMBED_DIMENSION', 1536))

    # ChromaDB持久化存储路径
    CHROMA_PERSIST_DIR = os.environ.get(
        'CHROMA_PERSIST_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_data')
    )

    # 文件上传配置
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 最大上传文件大小：50MB
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md', 'docx'}

    # 文档分块配置
    CHUNK_SIZE = 500        # 每个分块的字符数
    CHUNK_OVERLAP = 50      # 分块之间的重叠字符数

    # 向量化批处理配置
    EMBED_BATCH_SIZE = 10   # 每批发送给Ollama的分块数量
    EMBED_MAX_RETRIES = 3   # 嵌入失败最大重试次数

    # RAG检索配置
    RETRIEVER_TOP_K = 4     # 检索返回的相似文档数量
