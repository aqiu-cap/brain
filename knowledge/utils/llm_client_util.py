import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
def get_llm_client(response_format:bool=False):
    try:
        # 从.env获取连接llm需要数据
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        model_name = os.getenv("ITEM_MODEL")

        model_kwargs = {}
        if response_format:
            model_kwargs['response_format'] = {"type": "json_object"}

        client = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            # 其他参数，可以不写都有默认值
            temperature=0.1,
            extra_body={
                "enable_thinking": False,
            },
            # 强制LLM必须返回json格式
            model_kwargs= model_kwargs
        )
        return client
    except Exception as e:
        raise e
