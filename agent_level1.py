import anthropic

client = anthropic.Anthropic()

# ---- 工具实现（已给你，研究一下 calculate 为什么要过滤字符）----
def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "Error: 不允许的字符"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def finish(answer: str) -> str:
    return answer

# ==== 空 1：工具 Schema ====
# LLM 看不见你的 Python 函数，只能看见这份 JSON 描述。
# 为 calculate 和 finish 各写一个 schema（name/description/input_schema）。
tools = [
    # TODO: 你的代码
]

# ==== 空 2：工具分发器 ====
def execute_tool(name: str, inputs: dict) -> str:
    # TODO: 根据 name 调用对应函数，未知工具返回错误字符串
    pass

# ==== 空 3：核心循环（本关的灵魂）====
def run_agent(user_message: str, max_iterations: int = 10) -> str:
    messages = [{"role": "user", "content": user_message}]

    for i in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=messages,
            tools=tools,
        )

        # TODO: 根据 response.stop_reason 分支处理：
        #   - "tool_use": 遍历 response.content 找到 tool_use 块，
        #     执行工具，然后把【两条消息】追加进 messages：
        #       ① assistant 消息（内容是 response.content 本身）
        #       ② user 消息（内容是 tool_result，必须带 tool_use_id！）
        #     如果工具是 finish → 直接返回最终答案
        #   - "end_turn": LLM 直接给了文本，提取并返回
        pass

    return "⚠️ 达到最大迭代次数"

# ==== 空 4：跑起来 ====
if __name__ == "__main__":
    print(run_agent("What is (1847 * 23) + (512 / 4)?"))
