from pathlib import Path

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:

        self.logger.info("start")
        # 验证参数的有效性
        pdf_path, md_output_path = self.validate_param(state)


    def validate_param(self, state: ImportGraphState) -> ImportGraphState:

        pdf_path_obj = Path(pdf_path)

        pass