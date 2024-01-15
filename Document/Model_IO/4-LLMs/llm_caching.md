# 缓存
LangChain为LLMs提供了一个可选的缓存层。这有两个原因：

它可以节省您的钱，通过减少您向LLM提供商发出的API调用次数，如果您经常多次请求相同的completions。它可以加快您的应用程序的速度，通过减少您向LLM提供商发出的API调用次数。

```python
from langchain.globals import set_llm_cache
from langchain_openai import OpenAI

# To make the caching really obvious, lets use a slower model.
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", n=2, best_of=2)
```



```python
%%time
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

# The first time, it is not yet in cache, so it should take longer
llm.predict("Tell me a joke")
```



```text
CPU times: user 13.7 ms, sys: 6.54 ms, total: 20.2 ms
Wall time: 330 ms
```



```text
"\n\nWhy couldn't the bicycle stand up by itself? Because it was two-tired!"
```



```python
%%time
# The second time it is, so it goes faster
llm.predict("Tell me a joke")
```



```text
CPU times: user 436 µs, sys: 921 µs, total: 1.36 ms
Wall time: 1.36 ms
```



```text
"\n\nWhy couldn't the bicycle stand up by itself? Because it was two-tired!"
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
CPU times: user 29.3 ms, sys: 17.3 ms, total: 46.7 ms
Wall time: 364 ms
```



```text
'\n\nWhy did the tomato turn red?\n\nBecause it saw the salad dressing!'
```



```python
%%time
# The second time it is, so it goes faster
llm.predict("Tell me a joke")
```



```text
CPU times: user 4.58 ms, sys: 2.23 ms, total: 6.8 ms
Wall time: 4.68 ms
```



```text
'\n\nWhy did the tomato turn red?\n\nBecause it saw the salad dressing!'
```