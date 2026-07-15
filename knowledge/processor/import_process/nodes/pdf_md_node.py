import json
import os
import subprocess
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import PdfConversionError, FileProcessingError
from knowledge.processor.import_process.state import ImportGraphState

# pdf转换md节点类
class PdfToMdNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.logger.info("节点2：开始pdf转换md")
        # 1 参数校验
        # pdf文件存在，  pdf转换md输出目录有
        pdf_path, md_output_path = self.validate_param(state)

        # 2 使用subprocess创建子进程，执行minerU命令
        process_code = self.execute_mineru(pdf_path,md_output_path)
        # subprocess约定 成功返回0
        if process_code != 0:
            raise PdfConversionError("pdf转换失败")

        # 3 更新state数据
        md_path = self.get_md_path(pdf_path, md_output_path)
        state["md_path"] = md_path
        return state

    # 获取md转换之后路径
    def get_md_path(self,pdf_path, md_output_path):
        # 拼接路径
        # md_output_path / pdf文件名称（不带后缀） / auto / pdf文件名称.md
        # c:\a\b\abc.pdf
        #pdf_path.name
        file_name = pdf_path.stem
        md_path = md_output_path / file_name / "auto" / f"{file_name}.md"
        return str(md_path)

    #1 参数校验
    def validate_param(self,state: ImportGraphState
                        )->Tuple[Path, Path]:
        # 判断state是否有pdf路径
        # c:\dev\test\aaa.pdf
        import_file_path = state.get("import_file_path")
        # 判断pdf是否存在
        pdf_path_obj = Path(import_file_path)
        # 不存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(f"{import_file_path}不存在")

        # 获取md输出路径
        md_path = state.get("file_dir")
        if not md_path:
            # c:\a\b\abc.pdf --> c:\a\b
            md_path = pdf_path_obj.parent
        md_path_obj = Path(md_path)

        # 返回
        return pdf_path_obj, md_path_obj


    # 2 执行mineru
    def execute_mineru(self, pdf_path: Path, md_path: Path):

        self.logger.info("====开始转换md====")
        # 设置环境变量-指定mineru本地模型路径
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HOME"] = r"D:\know\mineru"
        os.environ["MODELSCOPE_CACHE"] = r"D:\know\mineru"

        # subprocess创建子进程执行 mineru命令
        cmd = [
            "mineru",
            "-p", str(pdf_path),
            "-o", str(md_path),
            "--source", "local"
            "--device", "cpu",
            "--backend", "pipeline",
            "--batch-size", "1",
            "--no-auto-download"
        ]
        proc = subprocess.Popen(
            cmd, # 执行mineru命令
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
        )

        for line in proc.stdout:
            self.logger.info(line.strip())

        # 等待完成
        process_code = proc.wait()
        if process_code == 0:
            self.logger.info("pdf sucess")
        else:
            self.logger.error("pdf error")
        return process_code


# if __name__ == "__main__":
#     # 初始化日志
#     setup_logging()
#
#     state = {
#         "import_file_path": r"D:\know\123.pdf"
#     }
#
#     node = PdfToMdNode()
#     res = node.process(state)
#     print(res)