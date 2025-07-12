# -*- coding: utf-8 -*-
# 华为考试数据处理脚本配置文件

# 默认模型配置
DEFAULT_MODEL_PATH = "/home/ma-user/work/Qwen2.5-3B-Instruct"

# 常用模型路径配置
MODEL_PATHS = {
    "qwen2.5-7b": "/home/ma-user/work/Qwen2.5-7B",
    "qwen2.5-7b-Instruct": "/home/ma-user/work/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "/home/ma-user/work/Qwen2.5-14B-Instruct", 
    "qwen2.5-3b-Instruct": "/home/ma-user/work/Qwen2.5-3B-Instruct", 
    "qwen2.5-3b": "/home/ma-user/work/Qwen2.5-3B",
     "btlm": "/home/ma-user/work/btlm-3b-8k-chat",
}

# 数据文件路径
DEFAULT_DATA_PATH = "./data/A2-data.jsonl"

# 生成参数配置
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": {
        "choice": 1024,
        "code-generate": 2048,
        "generic-generate": 1024,
        "math": 1024,
    }
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "encoding": "utf-8"
}

# 日志目录配置
LOGS_BASE_DIR = "/home/ma-user/work/logs"

# 统计文件路径配置
STATS_FILE_PATH = "/home/ma-user/work/performance_stats.csv"

# 需要trust_remote_code=True的模型配置
MODELS_REQUIRE_TRUST_REMOTE_CODE = {
    "btlm": True,
    "/home/ma-user/work/btlm-3b-8k-chat": True,
}

# 获取模型路径
# Args:
#     model_key (str): 模型键名，如果为None则返回默认模型路径
# Returns:
#     str: 模型路径
def get_model_path(model_key=None):
    if model_key is None:
        return DEFAULT_MODEL_PATH
    
    if model_key in MODEL_PATHS:
        return MODEL_PATHS[model_key]
    
    # 如果传入的是完整路径，直接返回
    if model_key.startswith("/"):
        return model_key
    
    # 如果找不到对应的模型键，返回默认模型
    print(f"警告: 未找到模型键 '{model_key}'，使用默认模型")
    return DEFAULT_MODEL_PATH

# 检查模型是否需要trust_remote_code=True
# Args:
#     model_key_or_path (str): 模型键名或完整路径
# Returns:
#     bool: 是否需要trust_remote_code=True
def needs_trust_remote_code(model_key_or_path):
    # 检查键名
    if model_key_or_path in MODELS_REQUIRE_TRUST_REMOTE_CODE:
        return MODELS_REQUIRE_TRUST_REMOTE_CODE[model_key_or_path]
    
    # 检查完整路径
    if model_key_or_path in MODEL_PATHS.values():
        for key, path in MODEL_PATHS.items():
            if path == model_key_or_path and key in MODELS_REQUIRE_TRUST_REMOTE_CODE:
                return MODELS_REQUIRE_TRUST_REMOTE_CODE[key]
    
    # 直接检查路径
    if model_key_or_path in MODELS_REQUIRE_TRUST_REMOTE_CODE:
        return MODELS_REQUIRE_TRUST_REMOTE_CODE[model_key_or_path]
    
    return False

# 列出所有可用的模型
def list_available_models():
    print("可用的模型:")
    for key, path in MODEL_PATHS.items():
        trust_note = " (需要trust_remote_code)" if needs_trust_remote_code(key) else ""
        print(f"  {key}: {path}{trust_note}")

# 清理日志文件
def clean_log_files():
    import os
    import shutil
    
    print("🧹 清理日志文件...")
    
    # 清理统计文件
    if os.path.exists(STATS_FILE_PATH):
        backup_file = STATS_FILE_PATH + '.backup'
        if os.path.exists(backup_file):
            os.remove(backup_file)
            print(f"  ✅ 已删除备份文件: {backup_file}")
        os.remove(STATS_FILE_PATH)
        print(f"  ✅ 已删除统计文件: {STATS_FILE_PATH}")
    
    # 清理日志目录
    if os.path.exists(LOGS_BASE_DIR):
        shutil.rmtree(LOGS_BASE_DIR)
        print(f"  ✅ 已删除日志目录: {LOGS_BASE_DIR}")
    
    print("🎉 日志文件清理完成！下次运行将从ID 001开始。") 