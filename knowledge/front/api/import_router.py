import os

import uvicorn
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Depends

from fastapi.responses import FileResponse

from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from knowledge.front.schema.task_schema import TaskStatusResponse
from knowledge.front.schema.upload_schema import UploadResponse
from knowledge.front.service.import_file_service import ImportFileService
from knowledge.front.service.task_service import TaskService
from knowledge.front.utils.deps import get_import_file_service, get_task_service
from knowledge.front.utils.paths import get_front_page_dir
from knowledge.processor.import_process.base import setup_logging


# fastAPI构建web服务，
# 接口路径 方法参数返回值
# 接口：访问页面
# 接口1：上传接口
# 接口2：根据task_id查询节点执行进度接口
def create_app() -> FastAPI:
    # 创建FastAPI对象
    app = FastAPI(description="知识库导入")

    # 跨域
    # app.add_middleware(
    #     CORSMiddleware,
    #     allow_origins=["*"],  # 允许任意的源
    #     allow_credentials=True,  # 允许cookie中携带任意的自定义参数
    #     allow_methods=["*"],  # 允许任意的请求方式
    #     allow_headers=["*"],  # 允许请求头中携带任意的我自定义参数
    # )

    # 加载静态资源，比如html css js等
    # 获取静态资源目录
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    # 注册路由
    register_router(app)
    return app

# 路由方法
def register_router(app: FastAPI):

    # 接口：访问页面
    @app.get("/import")
    async def import_page():
        return FileResponse(path=os.path.join(get_front_page_dir(), "import.html"))

    # 接口1：上传接口
    # 第一个参数：后台任务对象
    # 第二个参数：上传文件
    # 第三个参数：ImportFileService对象
    # -- taskservice = TaskService()
    # -- ImportFileService(taskservice)
    @app.post("/upload", response_model=UploadResponse)
    async def upload_file(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            service: ImportFileService=Depends(get_import_file_service)):

        # 1 把上传过来文件，存储到本地路径
        #         # task_id: 任务id
        #         # file_dir：本地保存上传文件目录
        #         # import_file_path：本地保存上传文件目录+文件名称
        task_id,file_dir,import_file_path = service.process_upload_file(file)

        # 2 创建后台任务，执行7个节点
        # abc(id,name)
        background_tasks.add_task(service.run_import_graph,
                                  task_id,file_dir,import_file_path)

        # 3 返回结果 task_id
        return UploadResponse(message="上传文件成功",task_id=task_id)


    # 接口2：根据task_id查询节点执行进度接口
    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status_endpoint(task_id: str,
                task_service: TaskService = Depends(get_task_service)):
        """
        根据任务id 查询任务的状态
        Returns:
        """
        task_info = task_service.get_task_info(task_id)
        return TaskStatusResponse(**task_info)

if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app=create_app(),port=8000,host="0.0.0.0")

