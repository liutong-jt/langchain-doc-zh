# Parent Document Retriever

当拆分文件以进行检索时，通常存在冲突的需求：

1. 您可能希望拥有较小的文档，以便其嵌入内容最准确地反映其含义。如果太长，嵌入内容可能会失去意义。
2. 您希望拥有足够长的文档，以便保留每个片段的上下文。

`ParentDocumentRetriever`通过拆分和存储小数据块来实现这种平衡。在检索过程中，它首先获取小块，然后查找这些块的父级 ID，并返回这些较大的文档。

请注意，“父级文档”指的是小块起源于的文档。这可以是整个原始文档，也可以是较大的块。

```python
from langchain.retrievers import ParentDocumentRetriever
```

```python
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
```

```python
loaders = [
    TextLoader("../../paul_graham_essay.txt"),
    TextLoader("../../state_of_the_union.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
```

## Retrieving full documents

在此模式下，我们希望检索完整的文档。因此，我们只指定一个子拆分器。

```python
# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)
```

```python
retriever.add_documents(docs, ids=None)
```

This should yield two keys，因为我们添加了两个文档。.

```python
list(store.yield_keys())
```

```text
['cfdf4af7-51f2-4ea3-8166-5be208efa040',
 'bf213c21-cc66-4208-8a72-733d030187e6']
```

让我们现在调用向量存储搜索功能 - 我们应该看到它返回小块（因为我们正在存储小块）。

```python
sub_docs = vectorstore.similarity_search("justice breyer")
```

```python
print(sub_docs[0].page_content)
```

```text
Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. 

One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.
```

现在让我们从整体检索器中检索。这应该会返回大量的文档 - 因为它返回的是包含较小片段的文档。

```python
retrieved_docs = retriever.get_relevant_documents("justice breyer")
```

```python
len(retrieved_docs[0].page_content)
```

```text
38540
```

## Retrieving larger chunks

有时，完整文档可能太大，以至于不希望按原样检索它们。在这种情况下，我们真正想做的是先将原始文档分割成较大的块，然后再将其分割成较小的块。然后我们对较小的块进行索引，但在检索时我们检索较大的块（但仍然不是完整的文档）。

```python
# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryStore()
```

```python
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

```python
retriever.add_documents(docs)
```

我们可以看到，现在远不止两个文件 - 这些是较大的片段。

```python
len(list(store.yield_keys()))
```

```text
66
```

让我们确保底层的向量存储仍然可以检索到小块。

```python
sub_docs = vectorstore.similarity_search("justice breyer")
```

```python
print(sub_docs[0].page_content)
```

```text
Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. 

One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.
```

```python
retrieved_docs = retriever.get_relevant_documents("justice breyer")
```

```python
len(retrieved_docs[0].page_content)
```

```text
1849
```

```python
print(retrieved_docs[0].page_content)
```

```text
In state after state, new laws have been passed, not only to suppress the vote, but to subvert entire elections. 

We cannot let this happen. 

Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. 

Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. 

One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. 

And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence. 

A former top litigator in private practice. A former federal public defender. And from a family of public school educators and police officers. A consensus builder. Since she’s been nominated, she’s received a broad range of support—from the Fraternal Order of Police to former judges appointed by Democrats and Republicans. 

And if we are to advance liberty and justice, we need to secure the Border and fix the immigration system. 

We can do both. At our border, we’ve installed new technology like cutting-edge scanners to better detect drug smuggling.  

We’ve set up joint patrols with Mexico and Guatemala to catch more human traffickers.  

We’re putting in place dedicated immigration judges so families fleeing persecution and violence can have their cases heard faster. 

We’re securing commitments and supporting partners in South and Central America to host more refugees and secure their own borders.
```

[
  ](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)