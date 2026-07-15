import base64
import json
import os
import re
from pathlib import Path
from typing import Tuple, List

from openai import OpenAI

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.minio_util import get_minio_client

# 图片处理节点
class MdImageNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.logger.info("节点3：开始md图片处理")
        #1 获取md路径 image路径 md内容
        md_path,image_path,md_content = self.get_path_conent(state)

        #2 判断图片是否存在
        if not image_path.exists():
            # 更新md_content
            state["md_content"] = md_content
            return state

        #3 从md内容里面找到图片位置，找到图片上文和下文，对上文和下文字符数量限制
        """
            [
            ('图片名称1','图片路径1',('图片前一个标题',"上文","下文")),
            ('图片名称2','图片路径3',('图片前一个标题',"上文","下文"))
            ]
        """
        img_context_data = self.get_img_context(image_path,md_content)

        #4 根据图片+ 上文 + 下文构建提示词，提供llm生成摘要
        # {"图片名称":"图片摘要"}
        summaries = self.get_summaries(img_context_data)

        #把md所有图片上传到minio服务，返回图片在minio路径
        # 更新md图片内容 ![llm摘要](minio路径)
        # 返回更新完成图片md内容
        new_md_content = self.upload_minio_update_mdcontent(
            md_path,
            md_content,
            summaries,
            img_context_data)

        state["md_content"] = new_md_content
        return state

    # #把md所有图片上传到minio服务，返回图片在minio路径
    #         # 更新md图片内容 ![llm摘要](minio路径)
    def upload_minio_update_mdcontent(self,md_path,
                                md_content,
                                summaries,
                                img_context_data):

        minio_urls = {}

        # c:\a\b\abc.md

        # 因为获取所有图片，图片路径 img_context_data
        # ('图片名称1','图片路径1',('图片前一个标题',"上文","下文")),
        for img_name,img_path,_ in img_context_data:
            # 获取每个图片路径
            image_path_local = str(img_path / img_name)

            # 使用minio上传
            minio_client = get_minio_client()

            name = md_path.stem
            # /a/a.jpg
            object_name = f"{name}/{img_name}"
            # 调用方法上传
            minio_client.fput_object(
                # bucket_name: str,  minio的bucket名称
                self.config.minio_bucket,
                # object_name: str, 上传文件在bucket路径和名称
                object_name,
                # file_path: str, 上传文件在本地路径
                image_path_local
            )
            #得到每个图片在minio路径
            # http://192.168.200.139:9000/knowledge-base-v2/abc/1.jpg
            minio_url = ("http://" + self.
                         config.minio_endpoint +
                         "/" + self.config.minio_bucket + "/" + object_name)
            # {"a.jpg":"http://...."}
            minio_urls[img_name] = minio_url

        # 把llm生成摘要 + 图片在minio路径更新md内容里面
        # 总流程： 得到一个图片，得到图片对应摘要 + minio路径
        #         拿着图片到md内容找到图片位置，更新图片摘要和路径
        #  summaries {"a.jpg":"这是一只老虎"}
        #  minio_urls {"a.jpg":"http://...."}
        new_md_content = md_content
        # 字典遍历
        for img_name,img_summary in summaries.items():
            # 根据图片名称获取minio路径
            img_url = minio_urls.get(img_name)

            # 到md内容更新图片信息
            replace_pattern = re.compile(
                r"!\[(.*?)\]\((.*?" + re.escape(img_name) + r".*?)\)",
                re.IGNORECASE)
            new_md_content = replace_pattern.sub(
                f"![{img_summary}]({img_url})",
                new_md_content
            )
        return new_md_content

    # 根据图片+ 上文 + 下文构建提示词，提供llm生成摘要
    # {"a.jpg":"这是一只老虎" ,"b.jpg":"这是一只野猪"}
    def get_summaries(self,img_context_data):
        # 定义空字典
        summaries = {}
        # 遍历上下文列表 img_context_data
        # img_path: 不带文件名称 D:\dev\6W100-整本手册\auto\images
        for img_name,img_path,img_context in img_context_data:
            # 得到每个图片信息，构建提示词，调用llm得到每个图片摘要数据
            image_path = str(img_path / img_name)
            # 构建提示词：
            single_summary = self.get_summary_llm(image_path,img_context)
            # 封装到summaries字典
            # {"图片名称":"图片摘要"}
            summaries[img_name] = single_summary
        return summaries

    # 根据数据构建提示词，调用llm生成图片摘要
    #  ('图片前一个标题',"上文","下文")
    def get_summary_llm(self,image_path,
                        img_context:Tuple[str,str,str]):
        # 解构
        front_title,pre_content,post_content = img_context

        # 把上面三个内容拼接字符串，作为提示词一部分
        context = []
        if front_title:
            context.append(front_title)
        if pre_content:
            context.append(pre_content)
        if post_content:
            context.append(post_content)
        final_context = "\n".join(context)

        # 把图片文件传递llm，把图片内容使用base64转换字符串
        # r 读  b 二进制类型 binary
        images_data_str = ""
        with open(image_path,"rb") as f:
            images_data_str = base64.b64encode(f.read()).decode("utf-8")

        # 创建连接llm客户端对象
        client = OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_api_base
        )

        # 构建提示词
        messages = [
            {
                "role":"user",
                "content":[
                    {
                        "type":"text",
                        "text":f"""任务：为Markdown文档中的图片生成一个简短的中文标题。
                         背景信息：
                             1. 图片上下文：{final_context}
                             请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片的内容，
                             生成一个精准的中文标题（不要包含"图片"二字）。""",
                    },
                    {
                        "type":"image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{images_data_str}"
                        }
                    }
                ]
            }
        ]

        # 调用llm得到结果
        response = client.chat.completions.create(
            model=self.config.vl_model,
            messages=messages,
        )
        summary = response.choices[0].message.content.strip()
        return summary

    # 从md内容里面找到图片位置，找到图片上文和下文，对上文和下文字符数量限制
    # [('图片名称1','图片路径1',('图片前一个标题',"上文","下文")),
    #   ('图片名称2','图片路径3',('图片前一个标题',"上文","下文"))]
    def get_img_context(self,image_path:Path,
           md_content:str)->List[Tuple[str,str,Tuple[str,str,str]]]:
        # 定义列表封装最终数据
        target_images_context = []

        # 从图片路径获取路径所有图片名称，
        # c:\a
        for image_name in os.listdir(image_path):
            # abc.jpg
            # 判断图片后缀名是否标准图片后缀名\
            # ("abc",".jpg")
            suffix = os.path.splitext(image_name)[1]
            if suffix not in self.config.image_extensions:
                continue

            # 拿着图片名称到md内容找到图片位置，图片上文和下文数据
            # [('图片前一个标题',"上文","下文")]
            img_context = self.get_img_title_pre_post(image_name,md_content)

            if not img_context:
                continue

            # 构建最终数据
            target_images_context.append((image_name, image_path,img_context[0]))

        return target_images_context

    #根据图片名称到md内容找到图片位置，图片上下文数据
    # 返回类型： [('图片前一个标题',"上文","下文")]
    # image_name: 图片名称   md_content： md内容
    def get_img_title_pre_post(self,image_name:str,
            md_content:str)->List[Tuple[str,str,str]]:
        # 总体思路：根据图片名称 到md内容找到图片位置，以找到图片为基准，
        # 找前一个标题 、上文 和 下文
        image_context_data=[]
        # 正则表达式：根据图片名称匹配标题使用
        re_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")

        # 把md_content  md内容根据\n拆分，得到md每行内容
        # md_lines 列表，放md每行内容
        md_lines = md_content.split("\n")

        # 图片前一个标题
        title = ""
        # 标题索引
        title_index = -1

        # md_lines列表遍历，得到每行内容，正则匹配
        # index 每行索引  line 每行内容
        for index,line in enumerate(md_lines):
            # 每行正则匹配,没有匹配到，继续下一行
            if not re_pattern.search(line):
                continue

            # 如果匹配图片  图片索引 index
            # 以图片为基准，找上文
            """
                0  # 标题  位置
                1
                2 
                3
                4
                5   图片  index
            """
            # 4 3 2 1 0 找到前一个标题为止
            for i in range(index-1,-1,-1):
                # 找到前一个标题为止
                if re.match(r"^#{1,6}\s+",md_lines[i]):
                    # 记录标题位置
                    title_index = i
                    # 获取标题
                    title = md_lines[i]
                    break

            # 图片 和 前一个标题之间内容 上文数据
            # [1,5)
            """   1 2 3 4
                           0  # 标题  位置
                           1
                           2 
                           3
                           4
                           5   图片  index
                       """
            pre_context = md_lines[title_index+1:index]
            # 调用方法对上文数据字符数量限制 不超过200
            final_pre_context = self.limit_context_number(
                pre_context,"front")

            """   1 2 3 4
                                      5   图片  index
                                      6
                                      7
                                      8
                                      9  # 标题  位置
                                      
             """
            # # 以图片为基准，找下文  abdc
            end_index = len(md_lines)
            for i in range(index+1,end_index):
                if re.match(r"^#{1,6}\s+",md_lines[i]):
                    # 下文标题位置
                    end_index = i
                    break
            # 图片 + 图片后一个标题之间数据是下文
            # [6,9)
            post_context = md_lines[index+1:end_index]
            # 调用方法对上文数据字符数量限制 不超过200
            final_post_context = self.limit_context_number(
                post_context,"back")

            # 封装最终数据
            # [('图片前一个标题',"上文","下文")]
            image_context_data.append((title,
                                       final_pre_context,
                                       final_post_context))
        return image_context_data

    # 限制上下文字符数量
    def limit_context_number(self,context,limit):
        # 不超过200
        max_char = 200
        # 列表
        final_context = []

        img_pattern = re.compile(r"^!\[.*?\]\(.*?\)$")
        # context列表遍历
        for line in context:
            # 获取不是空行 和 不是图片行数据
            md_line = line.strip()
            if md_line and not img_pattern.search(md_line):
                final_context.append(md_line)

        # 上文数据截取，从下往上截取切片
        """
            0  333
            1  222
            2  111
            3  图片
        """
        if limit == "front":
            # 333 222 111 => 111 222
            final_context.reverse()

        # 放小于200字符数据
        selected = []
        total = 0
        for line in final_context:
            # 遍历final_context得到每行数据
            # 计算每行数据字符数量
            size = len(line)
            if (total+size) > max_char:
                if not selected:
                    selected.append(line[:max_char])
                break
            selected.append(line)
            total += size

        # 上文，反转回来
        if limit == "front":
            selected.reverse()

        #selected列表 -- 字符串  join
        return "\n".join(selected)

    # 获取md路径 image路径 md内容
    def get_path_conent(self,state:ImportGraphState
                        )->Tuple[Path,Path,str]:
        # 获取md路径
        md_path = state.get("md_path")
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileNotFoundError(f"{md_path} does not exist")

        # 获取image路径
        image_path = md_path_obj.parent / "images"

        # 获取md内容 返回str
        with open(md_path_obj,"r",encoding="utf-8") as f:
            md_content = f.read()

        return md_path_obj,image_path,md_content

if __name__ == '__main__':
    state = {
        "md_path": r"D:\know\6W100-整本手册\auto\6W100-整本手册.md"
    }

    md_image_node = MdImageNode()

    res = md_image_node.process(state)
    print(json.dumps(res,ensure_ascii=False,indent=4))