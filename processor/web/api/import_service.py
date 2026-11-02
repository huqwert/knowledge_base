import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from processor.config.config import file_upload_config, minio_config
from processor.import_process.main_graph import KBImportWorkflow
from processor.utils.minio_utils import get_minio_client
from processor.utils.task_utils import add_running_task, add_done_task, get_running_task_list, get_done_task_list, \
    get_task_status, update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, \
    get_task_info
from processor.tool.logger import logger

# 1. 创建应用
app = FastAPI(
    title="导入API"
)

# 2. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 静态页面
@app.get("/import.html")
async def get_import_page():
    html_path = Path(__file__).absolute().parent.parent / "page" / "import.html"
    return FileResponse(html_path)

# 4. 后台任务
def run_graph_task(task_id, local_dir, local_file_path):

    try:
        # 1. 更新任务的全局状态为处理中
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        # 2. 定义初始化状态
        init_state = {
            "task_id": task_id,
            "local_dir": local_dir,
            "local_file_path": local_file_path
        }
        # 3. 启动工作流
        for chunk in KBImportWorkflow.create_and_run(init_state, stream=True):
            for node_name, node_result in chunk.items():
                # logger.info(f"{node_name}: {node_result}")
                pass

        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        update_task_status(task_id, TASK_STATUS_FAILED)
        logger.error(f"{task_id} : 任务执行失败: {e}", exc_info=True)
        # logger.exception(f"{task_id} : 任务执行失败: {e}")


# 5. 核心接口：文件上传接口
# 访问地址：http://localhost:8000/upload

@app.post("/upload", summary="文件上传接口", description="自动触发知识库导入全流程")
async def upload_files(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...,description="需要上传的文件（pdf、md）")
):
    # ======================将文件上传到Python服务器上====================
    # 1. 构建本地存储根目录：项目根目录/doc/YYYYMMDD（按日期分层，方便管理）
    data_based_root_dir = file_upload_config.data_based_root_dir
    data_dir = os.path.join(data_based_root_dir, datetime.now().strftime("%Y%m%d"))

    # 生成全局唯一TaskID（UUID4），作为单个文件的全流程标识
    task_id = str(uuid.uuid4())
    logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")

    # 3. 标记「文件上传」阶段为「运行中」，前端轮询可查
    add_running_task(task_id, "upload_file")

    # 4. 构建该任务的本地独立目录：output/YYYYMMDD/TaskID，避免多文件重名冲突
    local_dir = os.path.join(data_dir, task_id)
    os.makedirs(local_dir, exist_ok=True)  # 目录不存在则创建，存在则不做处理

    # 构建上传文件的本地保存绝对路径
    local_file_path = os.path.join(local_dir, file.filename)

    # 5. 将上传的文件保存到本地临时目录（后续MinIO上传/文件解析均基于此文件）
    with open(local_file_path, "wb") as file_buffer:
        # 默认1m
        shutil.copyfileobj(file.file, file_buffer, length=1024 * 1024)  # 1MB分块上传

    logger.info(f"[{task_id}] 文件已保存至本地，路径：{local_file_path}")

    # ======================将文件上传到MinIO服务器上====================
    try:
        minio_client = get_minio_client()
        minio_bucket_name = minio_config.bucket_name
        minio_object_name = f"pdf_files/{datetime.now().strftime('%Y%m%d')}/{file.filename}"

        minio_client.fput_object(minio_bucket_name, minio_object_name, local_file_path)
    except Exception as e:
        logger.warning(f"文件上传到MinIO失败: {e}")

    # 7. 对文件上传任务进行追踪(已完成)
    add_done_task(task_id,"upload_file")

    # 8. 启动后台任务,调用工作流
    background_tasks.add_task(run_graph_task, task_id, local_dir, local_file_path)

    return {
        "code": 200,
        "message": "文件上传成功",
        "task_id": task_id
    }

@app.get("/status/{task_id}", summary="任务状态查询接口", description="前端根据task_id对此接口进行定时轮询")
async def get_task_progress(task_id: str):


    task_status_info: Dict = get_task_info(task_id)

    return task_status_info

if __name__ == '__main__':
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )