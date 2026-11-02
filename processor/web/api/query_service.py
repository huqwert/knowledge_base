import uuid
from pathlib import Path

import uvicorn
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field, BaseModel
from starlette.responses import StreamingResponse

from processor.query_process.main_graph import KBQueryWorkflow
from processor.tool.logger import logger
from processor.utils.mongo_history_utils import clear_history, get_recent_messages
from processor.utils.sse_utils_sync import event_generator, create_sse_queue, push_progress
from processor.utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED
from fastapi import Request

# 1. 创建应用
app = FastAPI(
    title="查询API",
)

# 2. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


# 3. 静态页面路由
@app.get('/chat.html')
async def chat():
    html_path = Path(__file__).absolute().parent.parent / "page" / "chat.html"

    return FileResponse(html_path)

# 4. 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话ID")

# 5. 后台任务
def run_query_graph(session_id: str, task_id: str, user_query: str):

    try:
        # 1. 更新任务状态: 处理中
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        push_progress(task_id)

        # 2. 定义初始化状态
        init_state = {
            "original_query": user_query,
            "session_id": session_id,
            "task_id": task_id
        }

        # 3. 启动工作流(调用invoke)
        # KBQueryWorkflow.create_and_run(init_state, stream=True)
        for chunk in KBQueryWorkflow.create_and_run(init_state, stream=True):
            for node_name, node_result in chunk.items():
                logger.info(f"{node_name}: {node_result}")

        # 4. 更新任务状态: 完成
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        push_progress(task_id)

    except Exception as e:
        #  5. 更新任务状态:失败
        update_task_status(task_id, TASK_STATUS_FAILED)
        push_progress(task_id)
        logger.error(f"流程执行异常: {e}")

# 6. RAG查询
@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    # 1. 获取用户问题
    user_query = request.query

    # 2. 获取session_id,如果没有则创建一个
    session_id = request.session_id

    # 3. 生成任务id
    task_id = str(uuid.uuid4())

    # 4. 为当前task_id创建一个异步队列
    create_sse_queue(task_id)

    # 启动后台任务,调用工作流
    background_tasks.add_task(run_query_graph, session_id, task_id, user_query)

    # 返回结果
    return {
        "message": "查询请求已经提交，结果正在处理中...",
        "session_id": session_id,
        "task_id": task_id
    }


# 7. sse 实时返回结果
@app.get("/stream/{task_id}")
async def stream(task_id: str, request: Request):

    return StreamingResponse(
        event_generator(task_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )



# 8. 清空指定会话的历史记录
@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    count = clear_history(session_id)
    return {"message": "历史会话已清空", "deleted_count": count}


# 9. 查询指定会话的历史记录
@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    try:
        records = reversed(get_recent_messages(session_id, limit=limit))
        items = [{
            "_id": str(r.get("_id")) if r.get("_id") is not None else "",
            "session_id": r.get("session_id", ""),
            "role": r.get("role", ""),
            "text": r.get("text", ""),
            "rewritten_query": r.get("rewritten_query", ""),
            "item_names": r.get("item_names", []),
            "ts": r.get("ts")
        } for r in records]

        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")

# 10. 健康检查
@app.get("/health")
async def health():
    return {"ok": True}


if __name__ == '__main__':
    uvicorn.run(
        app,
        host='127.0.0.1',
        port=8001
    )