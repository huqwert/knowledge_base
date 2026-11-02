# atguigu/tool/download_bgem3.py

from modelscope.hub.snapshot_download import snapshot_download

# 下载模型到当前目录下的 models/bge-m3 文件夹
model_dir = snapshot_download('BAAI/bge-m3', cache_dir='E:/ai_models/modelscope_cache/models')
