import json
import os.path
import re
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.state import ImportGraphState

#文档切分节点
class DocumentSplitNode(BaseNode):
    def process(self,
         state: ImportGraphState)->ImportGraphState:

        self.logger.info("节点4：开始文档切分")
        # 1 获取md内容  file_title:标题默认值，不带后缀名文件名称
        md_content,file_title = self.get_data(state)
        print(f"第一个方法：获取数据 ")
        print("==" * 50)

        # 2 根据md内容，根据标题切分
        # parts 列表
        # [{body:11,title:标题1,parent_title:标题}]
        print(f"第二个方法：根据标题切分 ")
        parts = self.split_title_doc(md_content,file_title)
        print(f"根据标题切分之后结果： ")
        print(json.dumps(parts,ensure_ascii=False,indent=4))
        print("=="*50)

        # 3 把标题切分每段内容处理，如果内容过大再次切分，内容太小合并
        print(f"第三个方法：内容过大再次切分，内容太小合并 ")
        part_chunks = self.split_and_merge(parts)
        print(f"切分和合并之后结果： ")
        print(json.dumps(part_chunks, ensure_ascii=False, indent=4))
        print("----" * 50)

        # 后面逻辑满足实际业务场景，如果没有场景可以没有后面代码
        # 4 对上面切分和合并返回列表，根据业务需求重新组装，组装满足具体业务结构
        chunks = self.collect_data(part_chunks)
        print(f"最终组装结果：{chunks}")
        # 更新到state里面
        state['chunks'] = chunks
        # 备份json格式
        self.copy_chunks(chunks, state)
        return state

    # 备份  把chunks数据写入json文件里面
    def copy_chunks(self,chunks,state):

        file_dir = state.get('file_dir')
        # D:\dev\6W100-整本手册\auto\chunks.json
        output_path = os.path.join(file_dir,"chunks.json")

        with open(output_path,"w",encoding="utf-8") as f:
            json.dump(chunks,f,ensure_ascii=False,indent=4)

    # 满足业务数据重新组装
    def collect_data(self,final_chunks):
        chunks = []
        for chunk in final_chunks:

            title = chunk.get('title')
            body = chunk.get('body')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')

            # 最终内容 ：content 包含 title + body
            content = f"{title}\n\n{body}"

            data = {
                "title": title,
                "content": content,
                "file_title": file_title,
                "parent_title": parent_title,
            }
            chunks.append(data)
        return chunks

    # 3 把标题切分每段内容处理，如果内容过大再次切分，内容太小合并
    def split_and_merge(self,parts: List[Dict]):
        final_result = []
        # parts遍历
        for part in parts:
            # part标题切分每段内容
            # append原样追加  []
            """ part = [{1},{2}]
                final_result = []
                final_result.append(part)
                [[{1},{2}]]
            """
            """
                 part = [{1},{2}]
                final_result = []
                final_result.extend(part)
                [{1},{2}]
            """
            final_result.extend(self.split_large_part(part))
        # 内容太小合并
        final_res = self.merge_short_part(final_result)
        return final_res

    # 内容多大再切分
    # part: {body:11,title:标题1,parent_title:标题}
    def split_large_part(self,part):
        # 从part字典获取数据
        body = part.get("body")
        title = part.get("title")
        parent_title = part.get("parent_title")
        file_title = part.get("file_title")

        # 计算body大小
        size = len(body)
        # 如果body大小没有超过2000，返回
        if size < self.config.max_content_length:
            return [part]

        # 超过2000，再拆分，递归字符文件切分器
        # 创建切分器对象
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.max_content_length,
            chunk_overlap=0,
            separators=["\n\n","\n","。","！","？"," ",""],
            keep_separator=False,
        )
        # 调用切分器对象方法开始切分，返回列表
        texts = text_splitter.split_text(body)
        # 把texts遍历数据处理（满足业务需求，可以不做）
        final_result = []
        for index,text in enumerate(texts):
            final_result.append({
                "title": title+"_"+f"{index+1}",
                "parent_title":parent_title,
                "body":text,
                "file_title":file_title,
            })

        print(f"final_result: {final_result}")
        return final_result

    # 内容过小合并
    def merge_short_part(self,parts):
        max_char = self.config.max_content_length
        min_char = self.config.min_content_length
        # 从列表parts获取第一段内容
        current_part = parts[0]
        #最终数据列表
        filal_parts = []
        # 从第二段开始遍历
        for next_part in parts[1:]:
            # 判断current_part 和 next_part是同源
            same_parent = (current_part.get("parent_title")
                           == next_part.get("parent_title"))

            current_size = len(current_part.get("body"))
            if same_parent and current_size <= min_char:

                # 判断合并之后数据是否大于最大大小
                current_body = current_part.get("body")
                next_body = next_part.get("body")

                merge_size = len(current_body+"\n"+next_body)
                # 判断 大于最大不合并
                if merge_size > max_char:
                    # 当前段放到最终数据，进入下一个段数据
                    filal_parts.append(current_part)
                    current_part = next_part
                    continue
                # 合并
                current_part['body'] = (
                        current_part.get("body")+"\n"+next_part.get("body"))
                current_part['title'] = current_part["parent_title"]
            else:
                filal_parts.append(current_part)
                current_part = next_part
        # 处理最后一段内容
        filal_parts.append(current_part)
        return filal_parts


    # 2 根据md内容，根据标题切分
    # [{cotent:11,title:标题1,parent_title:标题}]
    def split_title_doc(self, md_content:str,
                        file_title:str)->List[Dict]:

        # md内容根据\n拆分，得到每行内容列表
        # md_lines列表
        md_lines = md_content.split("\n")

        # 遍历列表，得到每一行，是否标题
        # 原则：遇到标题结束，前面内容放到临时列表，合并称为一段
        temp = []  # 临时列表

        # 列表：最终数据
        res = []

        # 记录标题
        new_title = ""

        # ["","# ","## ","### ","","",""]
        level_title_list = [""] * 7
        # 列表索引值，对应哪一级标题，比如索引值1 对应一级标题
        current_level = 0  # 0 没有标题

        # 寻找标题正则表达式   *匹配0次或者多次    +匹配1次或者多次
        title_rule = re.compile(r"^\s*(#{1,6})\s+(.+)")

        for line in md_lines:
            # 找到标题
            if title_rule.match(line):
                # 遇到标题前面content临时列表合并
                content = "\n".join(temp).strip()
                # 找到当前合并这段标题，找到当前标题父标题
                # 拼接要求格式 {cotent:11,title:标题1,parent_title:标题}
                if new_title or content:
                    parent_title = ""
                    # current_level当前标题 在列表索引位置
                    # ["","# ","## ","### ","","",""]
                    for lv in range(current_level-1,0,-1):
                        if level_title_list[lv]:
                            parent_title = level_title_list[lv]
                            break

                    # parent_title没有找到默认值
                    if not parent_title:
                        parent_title = new_title if new_title else file_title

                    res.append({
                        "title": new_title,
                        "parent_title": parent_title,
                        "body": content,
                        "file_title": file_title
                    })

                #当前标题  存level_title_list    current_level
                # ["","# ","## ","### ","#### ","",""]

                title_obj = title_rule.match(line)
                if title_obj:
                    #  # 标题1   ## 标题2
                    # 获取#数量
                    level = len(title_obj.group(1))
                    current_level = level
                    # 放到level_title_list
                    level_title_list[level] = line

                    # 当前标题下面级别标题在列表清空
                    for lv in range(current_level+1,7,1):
                        level_title_list[lv] = ""

                # 记录标题
                new_title = line
                # 清空temp列表
                temp = []

            else: #不是标题
                temp.append(line)

        # 处理最后一部分数据，因为最后一行可能没有标题，按照前面逻辑处理不到
        # 最后在临时列表数据合并
        last_content = "\n".join(temp)
        if new_title or last_content:
            parent_title = ""

            for lv in range(current_level-1,0,-1):
                if level_title_list[lv]:
                    parent_title = level_title_list[lv]
                    break

            if not parent_title:
                parent_title = new_title if new_title else file_title

            res.append({
                "title": new_title,
                "parent_title": parent_title,
                "body": last_content,
                "file_title": file_title
            })
        return res

    # 1 获取md内容  file_title:标题默认值，不带后缀名文件名称
    def get_data(self,state:ImportGraphState):
        # 获取md内容
        md_content = state.get("md_content")
        if not md_content:
            raise Exception("md内容为空")
        # file_title
        file_title = state.get("file_title")
        return md_content,file_title

# if __name__ == "__main__":
#
#     # 构建md_content
#     file_path = r"D:\know\123\auto\123.md"
#
#     with open(file_path, "r", encoding="utf-8") as f:
#         file_content = f.read()
#
#     # 构建数据
#     state = {
#         "file_title": "123",
#         "md_content": file_content,
#         "file_dir": r"D:\know\test"
#     }
#     doc_split_node = DocumentSplitNode()
#     res = doc_split_node.process(state)
#     print(res)