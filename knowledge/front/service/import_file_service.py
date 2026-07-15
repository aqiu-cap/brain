
import os.path
import shutil
import uuid
from datetime import datetime
from typing import Tuple

from fastapi import UploadFile

from knowledge.front.service.task_service import TaskService
from knowledge.front.utils.paths import get_local_base_dir
from knowledge.processor.import_process.main_graph import run_graph_import


# 1 获取上传过来的文件
# 2 把上传过来文件存储本地目录
# 3 创建后台任务，调用langgraph执行7个节点
class ImportFileService:


    #1 初始化
    def __init__(self,task_service:TaskService):
        self.task_service = task_service

    #2  获取pdf保存路径
    def get_file_dir(self)->str:
        # get_local_base_dir() => ..knowledge/front/temp_data
        #  ..knowledge/front/temp_data/20260629
        return os.path.join(get_local_base_dir(),
                            datetime.now().strftime("%Y%m%d"))

    #3  保存上传文件到本地路径
    def save_upload_file_to_local(self,file:UploadFile,file_dir:str):
        # file_dir存在判断
        # c:\a\b\20260629
        os.makedirs(file_dir,exist_ok=True)
        # c:\a\b\20260629\test.pdf
        import_file_path = os.path.join(file_dir,file.filename)
        # 把上传file，写入到 import_file_path里面
        # file => UploadFile  .file
        with open(import_file_path,"wb") as f:
            # f.write(file.file.read())
            # 批量写入
            shutil.copyfileobj(file.file,f)
        return import_file_path

    #4 接口调用的方法
    def process_upload_file(self, file: UploadFile) ->Tuple[str,str,str]:
        # 获取保存上传文件本地路径
        # c:\a\b\20260629\
        date_dir = self.get_file_dir()
        # 生成任务id
        # 345678
        task_id = str(uuid.uuid4())
        # 构建最终保存上传文件本地路径
        # c:\a\b\20260629\task_id
        file_dir = os.path.join(date_dir,task_id)

        #1 记录当前正在运行节点，前端显示使用
        self.task_service.mark_node_running(task_id,"upload_file")

        #2 调用方法保存上传文件到本地
        import_file_path = self.save_upload_file_to_local(file,file_dir)

        #3 记录当前完成的节点，前端显示使用
        self.task_service.mark_node_done(task_id,"upload_file")

        #4 返回数据
        # task_id: 任务id
        # file_dir：本地保存上传文件目录
        # import_file_path：本地保存上传文件目录+文件名称
        return task_id,file_dir,import_file_path

    #5 langgraph执行节点的方法
    def run_import_graph(self, task_id: str, file_dir: str,
                         import_file_path: str):
        try:
            # 1 更新当前任务状态：任务处理中processing
            self.task_service.update_task_status(task_id,"processing")

            # 2 调用main_graph模块方法执行节点
            run_graph_import(task_id,import_file_path,file_dir)

            #3 导入多个节点执行完成，更新任务状态 completed
            self.task_service.update_task_status(task_id,"completed")

        except Exception as e:
            # 执行多个节点出错，任务状态修改为 failed
            self.task_service.update_task_status(task_id, "failed")
            print(f"{task_id} failed {e}")