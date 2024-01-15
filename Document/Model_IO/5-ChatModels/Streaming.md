# 流式处理

所有 ChatModel 都实现了 Runnable 接口，该接口带有所有方法的默认实现，即 ainvoke、batch、abatch、stream 和 astream。这为所有 ChatModel 提供了基本的流式处理支持。

流式处理默认返回一个迭代器（对于异步流式处理，返回 AsyncIterator），该迭代器返回底层 ChatModel 提供程序返回的单个值，即最终结果。这显然不支持按令牌流式处理，这需要 ChatModel 提供程序的本机支持，但确保期望迭代器的代码可以用于我们的任何 ChatModel 集成。

可以在以下位置查看 [支持按令牌流式处理的集成](https://python.langchain.com/docs/integrations/chat/)。

```python
from langchain_community.chat_models import ChatAnthropic
```



```python
chat = ChatAnthropic(model="claude-2")
for chunk in chat.stream("Write me a song about goldfish on the moon"):
    print(chunk.content, end="", flush=True)
```



```text
 Here's a song I just improvised about goldfish on the moon:

Floating in space, looking for a place 
To call their home, all alone
Swimming through stars, these goldfish from Mars
Left their fishbowl behind, a new life to find
On the moon, where the craters loom
Searching for food, maybe some lunar food
Out of their depth, close to death
How they wish, for just one small fish
To join them up here, their future unclear
On the moon, where the Earth looms
Dreaming of home, filled with foam
Their bodies adapt, continuing to last 
On the moon, where they learn to swoon
Over cheese that astronauts tease
As they stare back at Earth, the planet of birth
These goldfish out of water, swim on and on
Lunar pioneers, conquering their fears
On the moon, where they happily swoon
```