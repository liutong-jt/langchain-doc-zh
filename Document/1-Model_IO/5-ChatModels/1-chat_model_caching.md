# 缓存
LangChain 提供了一个可选的缓存层，用于聊天模型。这有两个原因：
它可以通过减少向 LLM 提供商发出的 API 调用次数来为您省钱，如果您经常多次请求相同的完成。它可以加快您的应用程序的速度，减少向 LLM 提供商发出的 API 调用次数。

```python
from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
```



## In Memory Cache

```python
%%time
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

# The first time, it is not yet in cache, so it should take longer
llm.predict("Tell me a joke")
```



```text
CPU times: user 17.7 ms, sys: 9.35 ms, total: 27.1 ms
Wall time: 801 ms
```



```text
"Sure, here's a classic one for you:\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!"
```



```python
%%time
# The second time it is, so it goes faster
llm.predict("Tell me a joke")
```



```text
CPU times: user 1.42 ms, sys: 419 µs, total: 1.83 ms
Wall time: 1.83 ms
```



```text
"Sure, here's a classic one for you:\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!"
```



## SQLite Cache

```python
!rm .langchain.db
```



```python
# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```



```python
%%time
# The first time, it is not yet in cache, so it should take longer
llm.predict("Tell me a joke")
```



```text
CPU times: user 23.2 ms, sys: 17.8 ms, total: 40.9 ms
Wall time: 592 ms
```



```text
"Sure, here's a classic one for you:\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!"
```



```python
%%time
# The second time it is, so it goes faster
llm.predict("Tell me a joke")
```



```text
CPU times: user 5.61 ms, sys: 22.5 ms, total: 28.1 ms
Wall time: 47.5 ms
```



```text
"Sure, here's a classic one for you:\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!"
```



[
  ](https://python.langchain.com/docs/modules/model_io/chat/quick_start)