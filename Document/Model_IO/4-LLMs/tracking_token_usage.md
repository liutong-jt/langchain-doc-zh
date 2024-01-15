# 跟踪令牌使用情况

本笔记本将介绍如何跟踪特定调用的令牌使用情况。目前，它仅适用于OpenAI API。

首先，让我们看一个非常简单的示例，演示如何为单个LLM调用跟踪令牌使用情况。

```python
from langchain.callbacks import get_openai_callback
from langchain_openai import OpenAI
```



```python
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", n=2, best_of=2)
```



```python
with get_openai_callback() as cb:
    result = llm.invoke("Tell me a joke")
    print(cb)
```



```text
Tokens Used: 37
    Prompt Tokens: 4
    Completion Tokens: 33
Successful Requests: 1
Total Cost (USD): $7.2e-05
```

在上下文管理器内部的任何内容都会被跟踪。以下是一个使用它来跟踪多个连续调用的示例。

```python
with get_openai_callback() as cb:
    result = llm.invoke("Tell me a joke")
    result2 = llm.invoke("Tell me a joke")
    print(cb.total_tokens)
```

```text
72
```

如果使用包含多个步骤的链或代理，它将跟踪所有这些步骤。

```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```

```python
with get_openai_callback() as cb:
    response = agent.run(
        "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?"
    )
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
```

```text
> Entering new AgentExecutor chain...
 I need to find out who Olivia Wilde's boyfriend is and then calculate his age raised to the 0.23 power.
Action: Search
Action Input: "Olivia Wilde boyfriend"
Observation: ["Olivia Wilde and Harry Styles took fans by surprise with their whirlwind romance, which began when they met on the set of Don't Worry Darling.", 'Olivia Wilde started dating Harry Styles after ending her years-long engagement to Jason Sudeikis — see their relationship timeline.', 'Olivia Wilde and Harry Styles were spotted early on in their relationship walking around London. (. Image ...', "Looks like Olivia Wilde and Jason Sudeikis are starting 2023 on good terms. Amid their highly publicized custody battle – and the actress' ...", 'The two started dating after Wilde split up with actor Jason Sudeikisin 2020. However, their relationship came to an end last November.', "Olivia Wilde and Harry Styles started dating during the filming of Don't Worry Darling. While the movie got a lot of backlash because of the ...", "Here's what we know so far about Harry Styles and Olivia Wilde's relationship.", 'Olivia and the Grammy winner kept their romance out of the spotlight as their relationship began just two months after her split from ex-fiancé ...', "Harry Styles and Olivia Wilde first met on the set of Don't Worry Darling and stepped out as a couple in January 2021. Relive all their biggest relationship ..."]
Thought: Harry Styles is Olivia Wilde's boyfriend.
Action: Search
Action Input: "Harry Styles age"
Observation: 29 years
Thought: I need to calculate 29 raised to the 0.23 power.
Action: Calculator
Action Input: 29^0.23
Observation: Answer: 2.169459462491557
Thought: I now know the final answer.
Final Answer: Harry Styles is Olivia Wilde's boyfriend and his current age raised to the 0.23 power is 2.169459462491557.

> Finished chain.
Total Tokens: 2205
Prompt Tokens: 2053
Completion Tokens: 152
Total Cost (USD): $0.0441
```