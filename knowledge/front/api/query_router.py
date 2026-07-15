import os
import asyncio
import threading
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  # 建议打开跨域，防止SSE被浏览器拦截

from knowledge.front.schema.query_schema import QueryRequest, StreamSubmitResponse, QueryResponse
from knowledge.front.service.query_service import QueryService
from knowledge.front.utils.deps import get_query_service
from knowledge.front.utils.paths import get_front_page_dir
from knowledge.utils.sse_util import sse_generator


def create_app() -> FastAPI:
    app = FastAPI(title="Query", description="知识库项目查询")

    # ✅ 强烈建议开启跨域，SSE 请求对 CORS 非常敏感
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)
    return app


def register_router(app: FastAPI):
    @app.get("/chat")
    async def get_chat():
        return FileResponse(os.path.join(get_front_page_dir(), "chat.html"))

    @app.post("/query")
    async def query(
            request: QueryRequest,
            background_tasks: BackgroundTasks,  # 这里保留参数，但我们不用它执行重任务
            service: QueryService = Depends(get_query_service),
    ):
        ###准备工作###
        session_id = request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()
        service.submit_query(task_id, request.is_stream)

        if request.is_stream:
            # ❌ 弃用 background_tasks.add_task，改用独立线程执行图任务
            # ✅ 这样可以100%保证不阻塞 FastAPI 主事件循环，让 /stream 接口能瞬间响应

            def run_graph_in_thread():
                """在独立线程中运行同步的图执行逻辑"""
                try:
                    # 如果 run_query_graph 是同步方法，直接调用
                    service.run_query_graph(task_id, session_id, request.query, True)
                except Exception as e:
                    print(f"[Router] ❌ 后台图执行异常: {e}", flush=True)

            # 🚀 启动独立守护线程
            thread = threading.Thread(target=run_graph_in_thread, daemon=True)
            thread.start()

            return StreamSubmitResponse(
                message="Query stream",
                session_id=session_id,
                task_id=task_id,
            )

        # 非流式 (保持原样)
        service.run_query_graph(task_id, session_id, request.query, False)
        answer = service.get_answer(task_id)
        return QueryResponse(
            message="Query",
            session_id=session_id,
            answer=answer,
        )

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        # 这个接口必须保持 async，且不能被 /query 的后台任务阻塞
        return StreamingResponse(
            sse_generator(task_id, request), media_type="text/event-stream",
        )

    # ... history 路由保持不变 ...
    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str, limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8001)