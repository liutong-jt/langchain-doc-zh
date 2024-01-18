# 流式处理

所有 `LLM` 都实现了 `Runnable` 接口，该接口提供了所有方法的默认实现，例如 ainvoke、batch、abatch、stream、astream。这为所有 `LLM` 提供了基本的
流式处理支持。

流式处理默认返回一个单值的迭代器（对于异步流式处理，返回 AsyncIterator）。这是由底层 `LLM` 提供商返回的最终结果。这显然不支持逐个令牌的流式处理，这需要 `LLM` 提供商的本机支持，但确保期望令牌迭代器的代码可以适用于我们的任何 `LLM` 集成。

在这里查看支持逐个令牌流式处理的 [集成](https://python.langchain.com/docs/integrations/llms/)。

```python
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=512)
for chunk in llm.stream("Write me a song about sparkling water."):
    print(chunk, end="", flush=True)
```



```text
Verse 1:
Bubbles dancing in my glass
Clear and crisp, it's such a blast
Refreshing taste, it's like a dream
Sparkling water, you make me beam

Chorus:
Oh sparkling water, you're my delight
With every sip, you make me feel so right
You're like a party in my mouth
I can't get enough, I'm hooked no doubt

Verse 2:
No sugar, no calories, just pure bliss
You're the perfect drink, I must confess
From lemon to lime, so many flavors to choose
Sparkling water, you never fail to amuse

Chorus:
Oh sparkling water, you're my delight
With every sip, you make me feel so right
You're like a party in my mouth
I can't get enough, I'm hooked no doubt

Bridge:
Some may say you're just plain water
But to me, you're so much more
You bring a sparkle to my day
In every single way

Chorus:
Oh sparkling water, you're my delight
With every sip, you make me feel so right
You're like a party in my mouth
I can't get enough, I'm hooked no doubt

Outro:
So here's to you, my dear sparkling water
You'll always be my go-to drink forever
With your effervescence and refreshing taste
You'll always have a special place.
```