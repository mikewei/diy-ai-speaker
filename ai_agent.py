# main.py
from fastapi import FastAPI
from langchain_openai import ChatOpenAI # 使用兼容OpenAI的LangChain LLM
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# 假设你的vLLM服务器运行在 http://localhost:8000
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "qwen3-14b"

app = FastAPI(title="MCP Agent Server")

# 定义LLM（通过LiteLLM或直接ChatOpenAI指向vLLM）
# Option 1: 直接使用 ChatOpenAI (vLLM的OpenAI兼容API)
llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key="EMPTY", # type:ignore # vLLM不需要实际的key
    openai_api_base=VLLM_BASE_URL, # type:ignore
    temperature=0.7,
    max_tokens=512 # type:ignore
)

# Option 2: 使用 LiteLLM 统一接口 (更灵活，如果未来有更多模型)
# from litellm import completion
# def vllm_llm_call(messages, temperature, max_tokens):
#     response = completion(
#         model=MODEL_NAME,
#         messages=messages,
#         api_base=VLLM_BASE_URL,
#         temperature=temperature,
#         max_tokens=max_tokens
#     )
#     return response.choices[0].message.content
# # 然后将这个函数包装成LangChain的Callable模型

# 示例工具（如果MCP需要外部工具调用）
def get_current_weather(location: str):
    """Get the current weather in a given location."""
    # 实际这里会调用外部天气API
    if "beijing" in location.lower():
        return "北京：25摄氏度，晴天"
    else:
        return "未知天气数据"

tools = [
    Tool(
        name="get_current_weather",
        func=get_current_weather,
        description="查询当前天气信息。输入参数：地点（字符串）。"
    )
]

# 定义Agent的Prompt (ReAct 风格的Prompt是MCP的常见模式)
# 这是一个简化版的Prompt，真实MCP可能更复杂
prompt = PromptTemplate.from_template(
    """
    你是一个智能助手，能够回答问题并执行工具。

    你可以使用的工具：
    {tools}

    使用以下格式回答：

    问题：用户的问题
    思考：你需要思考什么？
    工具：使用的工具名称 (如果有)
    工具输入：工具的输入 (如果有)
    观察：工具的输出 (如果有)
    ...（重复思考/工具/工具输入/观察直到你得到最终答案）
    思考：思考你如何回答
    答案：最终答案

    问题：{input}
    思考：
    """
)


# 创建Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# 定义API端点来接收MCP请求
@app.post("/mcp/chat")
async def chat_with_agent(
    query: str,
    history: list[dict] = [] # 假设历史是 {"role": "user/assistant", "content": "..."}
):
    # 转换历史记录为LangChain Messages
    langchain_history = []
    for msg in history:
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_history.append(AIMessage(content=msg["content"]))

    # 在AgentExecutor中执行
    try:
        # LangChain AgentExecutor 内部会处理历史和工具调用
        # 对于create_react_agent，通常不需要显式传递history到AgentExecutor的run方法
        # 而是将其融合到Prompt中，或者在AgentExecutor内部通过Memory来管理
        # 这里简化为直接传入input
        result = agent_executor.invoke({"input": query, "chat_history": langchain_history})
        return {"response": result["output"]}
    except Exception as e:
        return {"error": str(e)}, 500

# 启动FastAPI服务器
# python -m uvicorn ai_speaker_agent:app --host 0.0.0.0 --port 8001 --reload