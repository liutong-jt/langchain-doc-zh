# Time-weighted vector store retriever

这个检索器使用了语义相似性和时间衰减的组合。 

对它们进行评分的算法是：

```text
semantic_similarity + (1.0 - decay_rate) ^ hours_passed
```

值得注意的是，`hours_passed`指的是检索器中对象上次被访问以来经过的小时数，而不是自创建以来的小时数。这意味着频繁访问的对象会保持“新鲜”。

```python
from datetime import datetime, timedelta

import faiss
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
```

## Low decay rate

一个低`衰减率`（在这里，为了达到极致，我们将它设置接近0）意味着记忆将会被“记住”更长时间。`衰减率`为0意味着记忆永远不会被遗忘，使得这个检索器等同于向量查找。

```python
# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore, decay_rate=0.0000000000000000000000001, k=1
)
```

```python
yesterday = datetime.now() - timedelta(days=1)
retriever.add_documents(
    [Document(page_content="hello world", metadata={"last_accessed_at": yesterday})]
)
retriever.add_documents([Document(page_content="hello foo")])
```

```text
['c3dcf671-3c0a-4273-9334-c4a913076bfa']
```

```python
# "Hello World" is returned first because it is most salient, and the decay rate is close to 0., meaning it's still recent enough
retriever.get_relevant_documents("hello world")
```

```text
[Document(page_content='hello world', metadata={'last_accessed_at': datetime.datetime(2023, 12, 27, 15, 30, 18, 457125), 'created_at': datetime.datetime(2023, 12, 27, 15, 30, 8, 442662), 'buffer_idx': 0})]
```



## High decay rate

具有高`衰减率`（例如，多个9）的`最近度分数`会迅速变为0！如果您将此设置为1，则所有对象的`最近度`均为0，再次使其等效于向量查找。

```python
# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore, decay_rate=0.999, k=1
)
```

```python
yesterday = datetime.now() - timedelta(days=1)
retriever.add_documents(
    [Document(page_content="hello world", metadata={"last_accessed_at": yesterday})]
)
retriever.add_documents([Document(page_content="hello foo")])
```

```text
['eb1c4c86-01a8-40e3-8393-9a927295a950']
```

```python
# "Hello Foo" is returned first because "hello world" is mostly forgotten
retriever.get_relevant_documents("hello world")
```

```text
[Document(page_content='hello foo', metadata={'last_accessed_at': datetime.datetime(2023, 12, 27, 15, 30, 50, 57185), 'created_at': datetime.datetime(2023, 12, 27, 15, 30, 44, 720490), 'buffer_idx': 1})]
```

## Virtual time

使用LangChain中的一些工具，你可以模拟时间组件。

```python
import datetime

from langchain.utils import mock_now
```

```python
# Notice the last access time is that date time
with mock_now(datetime.datetime(2024, 2, 3, 10, 11)):
    print(retriever.get_relevant_documents("hello world"))
```

```text
[Document(page_content='hello world', metadata={'last_accessed_at': MockDateTime(2024, 2, 3, 10, 11), 'created_at': datetime.datetime(2023, 12, 27, 15, 30, 44, 532941), 'buffer_idx': 0})]
```
