import datetime
from knowledge.front.utils.task_util import set_task_result
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import ANSWER_PROMPT
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.mongo_client_util import save_chat_message
from knowledge.utils.sse_util import push_sse_event, SSEEvent

# 答案生成节点
class AnswerOutputNode(BaseNode):
    def process(self,state:QueryGraphState)->QueryGraphState:
        #1 获取state里面是否有answer数据，如果有，直接返回
        # 没有answer → 构建提示词调用LLM生成
        if not state.get("answer"):
            # 2.1 构建提示词
            prompt = self.build_prompt(state)
            state["prompt"] = prompt

            # 2.2 调用LLM生成答案
            self.call_llm_generate_answer(prompt,state)

            # 2.3 存储历史会话记录到MongoDB
            self.save_history_to_mongodb(state)

        # 统一写入全局任务结果（非流式 get_answer 需要）
        set_task_result(state.get("task_id"),
                        "answer",state.get("answer"))

        # 流式 → SSE 推送最终结果
        if state.get("is_stream"):
            push_sse_event(state.get("task_id"),SSEEvent.FINAL,
                           {"answer":state.get("answer")})

        return state

    #1 构建这些内容构建提示词
    # 问题 + item_name + 历史会话 + reranker之后结果
    def build_prompt(self,state:QueryGraphState)->str:
        # 问题
        rewritten_query = state.get("rewritten_query")
        # item_name
        item_name = state.get("item_name")
        # reranker之后结果
        reranked_docs = state.get("reranked_docs")
        # 历史会话
        history = state.get("history")

        # reranker操作之后获取结果，拼接字符串
        reranked_docs_str = self.build_rerankered_result(reranked_docs)

        # 历史会话获取内容，拼接字符串
        history_str = self.build_history_result(history)

        prompt = ANSWER_PROMPT.format(
            context=reranked_docs_str,
            history=history_str if history_str else "暂无历史会话",
            item_names=item_name,
            question=rewritten_query)
        today = datetime.date.today().strftime("%Y年%m月%d日")
        prompt = f"当前日期：{today}\n\n" + prompt
        return prompt

    def build_rerankered_result(self,reranked_docs)->str:
        result = []
        total = 0
        for index,doc in enumerate(reranked_docs):
            content = doc.get("content")

            # 拼接字符串
            other = {
                "title":doc.get("title",""),
                "url":doc.get("url","")
            }
            data = f"{index}:{other}\n{content}"

            result.append(data)
            total += len(data)

            if total + len(data) > 2000:
                break

        return "\n".join(result)

    def build_history_result(self,history)->str:
        result = []
        used = 0
        for message in history:
            text = message.get('text', '')
            role = message.get('role', '')

            # user:1111
            data = f"{role}:{text}"

            if used + len(data) > 2500:
                break

            result.append(data)
            used += len(data)
        return "\n".join(result)

    #2 调用llm，得到结果，流式 非流式 ，sse事件
    def call_llm_generate_answer(self,prompt:str,state:QueryGraphState):
        # 获取llm连接对象
        llm_client = get_llm_client()
        # 获取流式 非流式输出
        is_stream = state.get("is_stream")
        task_id = state.get("task_id")
        if is_stream: # 流式
            stream_result = self.stream_output(llm_client,prompt,task_id)
            # 更新到state
            state["answer"] = stream_result
        else: # 非流式
            invoke_result = self.invoke_output(llm_client,prompt)
            # 更新到state
            state["answer"] = invoke_result

    # 流式
    def stream_output(self,llm_client,prompt,task_id):
        final_answer = ""
        # 调用stream方法一段一段内容返回，遍历得到一段一段内容
        for chunk in llm_client.stream(prompt):
            delta_text = getattr(chunk,"content")
            if delta_text:
                final_answer += delta_text
                # 推送给前端，使用sse技术
                push_sse_event(task_id,SSEEvent.DELTA,
                               {"delta":delta_text})
        return final_answer

    # 非流式
    def invoke_output(self,llm_client,prompt):
        result = llm_client.invoke(prompt)
        return result.content

    #3 保存历史会话记录到MongoDB
    def save_history_to_mongodb(self,state):
        # 保存用户问题
        save_chat_message(
            session_id=state.get("session_id"),
            role="user",
            text=state.get("original_query"),
            rewritten_query=state.get("rewritten_query"),
            item_names=state.get("item_names"),
        )

        # 保存问题的答案
        save_chat_message(
            session_id=state.get("session_id"),
            role="assistant",
            text=state.get("answer"),
            rewritten_query=state.get("rewritten_query"),
            item_names=state.get("item_names"),
        )


