# Ensemble Retriever

`EnsembleRetriever` 输入是一个检索器列表，它将这些检索器的 `get_relevant_documents()` 方法的结果进行整合，并根据 [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) 算法对结果进行重新排序。

通过利用不同算法的优点，`EnsembleRetriever` 可以获得比任何单一算法更好的性能。

最常见的模式是将稀疏检索器（如 BM25）与密集检索器（如嵌入相似性）结合在一起，因为它们的优点是互补的。这也被称为“混合搜索”。稀疏检索器擅长根据关键词找到相关文档，而密集检索器擅长根据语义相似性找到相关文档。

```python
%pip install --upgrade --quiet  rank_bm25 > /dev/null
```

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
```

```python
doc_list_1 = [
    "I like apples",
    "I like oranges",
    "Apples and oranges are fruits",
]

# initialize the bm25 retriever and faiss retriever
bm25_retriever = BM25Retriever.from_texts(
    doc_list_1, metadatas=[{"source": 1}] * len(doc_list_1)
)
bm25_retriever.k = 2

doc_list_2 = [
    "You like apples",
    "You like oranges",
]

embedding = OpenAIEmbeddings()
faiss_vectorstore = FAISS.from_texts(
    doc_list_2, embedding, metadatas=[{"source": 2}] * len(doc_list_2)
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)
```

```python
docs = ensemble_retriever.invoke("apples")
docs
```

```text
[Document(page_content='You like apples', metadata={'source': 2}),
 Document(page_content='I like apples', metadata={'source': 1}),
 Document(page_content='You like oranges', metadata={'source': 2}),
 Document(page_content='Apples and oranges are fruits', metadata={'source': 1})]
```

## Runtime Configuration

We can also configure the retrievers at runtime. In order to do this, we need to mark the fields as configurable

```python
from langchain_core.runnables import ConfigurableField
```



```python
faiss_retriever = faiss_vectorstore.as_retriever(
    search_kwargs={"k": 2}
).configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs_faiss",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)
```



```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)
```



```python
config = {"configurable": {"search_kwargs_faiss": {"k": 1}}}
docs = ensemble_retriever.invoke("apples", config=config)
docs
```



Notice that this only returns one source from the FAISS retriever, because we pass in the relevant configuration at run time

[
  ](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression)