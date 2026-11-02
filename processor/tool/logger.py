# atguigu/tool/logger.py

import logging
import colorlog

# 获取日志记录器
logger = logging.getLogger()
# 设置日志记录器的日志级别
logger.setLevel(logging.INFO)

# 设置日志处理器
handler = colorlog.StreamHandler()
# 设置格式
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

# logger.handlers.clear()
logger.addHandler(handler)