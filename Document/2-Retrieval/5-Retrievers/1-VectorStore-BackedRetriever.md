# 向量存储检索器

向量存储检索器是使用向量存储检索文档的检索器。它是一个轻量级的包装器，用于使向量存储类符合检索器接口。它使用向量存储实现的搜索方法，如相似度搜索和MMR，来查询向量存储中的文本。

构建一个向量存储后，构建一个检索器非常容易。让我们通过一个例子来说明。

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../state_of_the_union.txt")
```

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
```

```python
retriever = db.as_retriever()
```

```python
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```

## 最大边际相关性检索 

默认情况下，向量存储检索器使用相似性搜索。如果底层向量存储支持最大边际相关性搜索，则可以指定其为搜索类型。

```python
retriever = db.as_retriever(search_type="mmr")
```

```python
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```



## 相似度分数阈值检索 

您还可以设置一种检索方法，该方法设置一个相似度分数阈值，并且只返回分数高于该阈值的文档。

```python
retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)
```

```python
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```



## Specifying top k

你也可以指定检索时使用的搜索参数，如 `k`。

```python
retriever = db.as_retriever(search_kwargs={"k": 1})
```

```python
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
len(docs)
```

```text
1
```