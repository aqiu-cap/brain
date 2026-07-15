from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import ImportProcessError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState

# 节点1：文件类型检测
class EntryNode(BaseNode):
    # 实现父类里面抽象方法 process方法
    # 写核心逻辑
    def process(self, state: ImportGraphState) -> ImportGraphState:

        self.logger.info("节点1：开始文件类型检测")

        # 获取上一步传递文件路径
        import_file_path = state.get('import_file_path')

        # 从这个路径获取后缀名
        path_obj = Path(import_file_path)

        # 获取后缀名  .pdf
        suffix = path_obj.suffix

        # 判断后缀名pdf  md
        # 根据不同类型更新不同state数据
        if suffix == '.pdf':
            # is_pdf_read_enabled true
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path
            self.logger.info("当前文件类型：pdf")
        elif suffix == '.md':
            # is_md_read_enabled
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
            self.logger.info("当前文件类型：md")
        else:
            self.logger.warn("当前文件类型其他类型")
            raise ValidationError("文件类型不匹配")

        # 获取文件标题  不带后缀名文件名称
        # c:\abc\img\test.pdf
        file_title = path_obj.stem
        state["file_title"] = file_title
        return state

# if __name__ == '__main__':
#     # 初始化日志
#     setup_logging()
#
#     # 构建state数据
#     state = {
#         "import_file_path": r"D:\know\万用表的使用.pdf"
#     }
#
#     # 创建对象
#     node = EntryNode()
#
#     # 调用process方法
#     res = node.process(state)
#     print(res)
