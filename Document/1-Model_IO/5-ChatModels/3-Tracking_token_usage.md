# 跟踪令牌使用情况

本笔记本将介绍如何为特定调用跟踪令牌使用情况。目前，它仅针对OpenAI API实现。

首先，让我们看一下跟踪单个聊天模型调用的令牌使用情况的极其简单的示例。

```python
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
```



```python
llm = ChatOpenAI(model_name="gpt-4")
```



```python
with get_openai_callback() as cb:
    result = llm.invoke("Tell me a joke")
    print(cb)
```



```text
Tokens Used: 24
    Prompt Tokens: 11
    Completion Tokens: 13
Successful Requests: 1
Total Cost (USD): $0.0011099999999999999
```



在上下文管理器内的任何内容都会被跟踪。以下是一个使用它来跟踪多个顺序调用的例子。

```python
with get_openai_callback() as cb:
    result = llm.invoke("Tell me a joke")
    result2 = llm.invoke("Tell me a joke")
    print(cb.total_tokens)
```



```text
48
```



If a chain or agent with multiple steps in it is used, it will track all those steps.

```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI

tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
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

Invoking: `Search` with `Olivia Wilde's current boyfriend`


['Things are looking golden for Olivia Wilde, as the actress has jumped back into the dating pool following her split from Harry Styles — read ...', "“I did not want service to take place at the home of Olivia's current partner because Otis and Daisy might be present,” Sudeikis wrote in his ...", "February 2021: Olivia Wilde praises Harry Styles' modesty. One month after the duo made headlines with their budding romance, Wilde gave her new beau major ...", 'An insider revealed to People that the new couple had been dating for some time. "They were in Montecito, California this weekend for a wedding, ...', 'A source told People last year that Wilde and Styles were still friends despite deciding to take a break. "He\'s still touring and is now going ...', "... love life. “He's your typical average Joe.” The source adds, “She's not giving too much away right now and wants to keep the relationship ...", "Multiple sources said the two were “taking a break” from dating because of distance and different priorities. “He's still touring and is now ...", 'Comments. Filed under. celebrity couples · celebrity dating · harry styles · jason sudeikis · olivia wilde ... Now Holds A Darker MeaningNYPost.', '... dating during filming. The 39-year-old did however look very cosy with the comedian, although his relationship status is unknown. Olivia ...']
Invoking: `Search` with `Harry Styles current age`
responded: Olivia Wilde's current boyfriend is Harry Styles. Let me find out his age for you.

29 years
Invoking: `Calculator` with `29 ^ 0.23`


Answer: 2.169459462491557Harry Styles' current age (29 years) raised to the 0.23 power is approximately 2.17.

> Finished chain.
Total Tokens: 1929
Prompt Tokens: 1799
Completion Tokens: 130
Total Cost (USD): $0.06176999999999999
```