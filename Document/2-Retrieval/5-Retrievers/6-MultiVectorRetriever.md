# MultiVector Retriever

存储每个文档中的多个向量通常是有益的。这在多个用例中都是有益的。LangChain有一个基本的`MultiVectorRetriever`，可以轻松查询此类设置。很多复杂性在于如何为每个文档创建多个向量。这个笔记本介绍了一些创建这些向量并使用`MultiVectorRetriever`的常用方法。

为每个文档创建多个向量的方法包括：

- 较小的片段：将文档拆分为较小的片段，并嵌入这些片段（这是ParentDocumentRetriever）。
- 摘要：为每个文档创建一个摘要，与（或代替）文档一起嵌入。
- 假设性问题：为每个文档创建假设性问题，每个文档都适合回答，与（或代替）文档一起嵌入。

请注意，这还启用了一种添加嵌入的另一种方法 - 手动。这很好，因为您可以显式添加应导致恢复文档的问题或查询，从而为您提供更多控制。

```python
from langchain.retrievers.multi_vector import MultiVectorRetriever
```

```python
from langchain.storage import InMemoryByteStore
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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
docs = text_splitter.split_documents(docs)
```

## Smaller chunks

很多时候，检索大量信息并嵌入小块信息可能会很有用。这使得嵌入可以尽可能紧密地捕获语义含义，但尽可能多地将上下文传递到下游。请注意，这就是`ParentDocumentRetriever`所做的。这里我们展示了幕后发生的情况。

```python
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
import uuid

doc_ids = [str(uuid.uuid4()) for _ in docs]
```

```python
# The splitter to use to create smaller chunks
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
```

```python
sub_docs = []
for i, doc in enumerate(docs):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)
```



```python
retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
```



```python
# Vectorstore alone retrieves the small chunks
retriever.vectorstore.similarity_search("justice breyer")[0]
```



```text
Document(page_content='Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. \n\nOne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.', metadata={'doc_id': '2fd77862-9ed5-4fad-bf76-e487b747b333', 'source': '../../state_of_the_union.txt'})
```

```python
# Retriever returns larger chunks
len(retriever.get_relevant_documents("justice breyer")[0].page_content)
```

```text
9875
```

翻译：检索器在向量数据库上执行的默认搜索类型是相似性搜索。LangChain 向量存储也支持通过 [最大边际相关性](https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html#langchain_core.vectorstores.VectorStore.max_marginal_relevance_search) 进行搜索，因此如果您想要使用这种方式，只需将 `search_type` 属性设置为如下：

```python
from langchain.retrievers.multi_vector import SearchType

retriever.search_type = SearchType.mmr

len(retriever.get_relevant_documents("justice breyer")[0].page_content)
```



```text
9875
```



## Summary[](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector#summary)

Oftentimes a summary may be able to distill more accurately what a chunk is about, leading to better retrieval. Here we show how to create summaries, and then embed those.

```python
import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
```



```python
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | ChatOpenAI(max_retries=0)
    | StrOutputParser()
)
```



```python
summaries = chain.batch(docs, {"max_concurrency": 5})
```



```python
# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]
```



```python
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]
```



```python
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
```



```python
# # We can also add the original chunks to the vectorstore if we so want
# for i, doc in enumerate(docs):
#     doc.metadata[id_key] = doc_ids[i]
# retriever.vectorstore.add_documents(docs)
```



```python
sub_docs = vectorstore.similarity_search("justice breyer")
```



```python
sub_docs[0]
```



```text
Document(page_content="The document is a speech given by President Biden addressing various issues and outlining his agenda for the nation. He highlights the importance of nominating a Supreme Court justice and introduces his nominee, Judge Ketanji Brown Jackson. He emphasizes the need to secure the border and reform the immigration system, including providing a pathway to citizenship for Dreamers and essential workers. The President also discusses the protection of women's rights, including access to healthcare and the right to choose. He calls for the passage of the Equality Act to protect LGBTQ+ rights. Additionally, President Biden discusses the need to address the opioid epidemic, improve mental health services, support veterans, and fight against cancer. He expresses optimism for the future of America and the strength of the American people.", metadata={'doc_id': '56345bff-3ead-418c-a4ff-dff203f77474'})
```



```python
retrieved_docs = retriever.get_relevant_documents("justice breyer")
```



```python
len(retrieved_docs[0].page_content)
```



```text
9194
```



## Hypothetical Queries[](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector#hypothetical-queries)

An LLM can also be used to generate a list of hypothetical questions that could be asked of a particular document. These questions can then be embedded

```python
functions = [
    {
        "name": "hypothetical_questions",
        "description": "Generate hypothetical questions",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["questions"],
        },
    }
]
```



```python
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser

chain = (
    {"doc": lambda x: x.page_content}
    # Only asking for 3 hypothetical questions, but this could be adjusted
    | ChatPromptTemplate.from_template(
        "Generate a list of exactly 3 hypothetical questions that the below document could be used to answer:\n\n{doc}"
    )
    | ChatOpenAI(max_retries=0, model="gpt-4").bind(
        functions=functions, function_call={"name": "hypothetical_questions"}
    )
    | JsonKeyOutputFunctionsParser(key_name="questions")
)
```



```python
chain.invoke(docs[0])
```



```text
["What was the author's first experience with programming like?",
 'Why did the author switch their focus from AI to Lisp during their graduate studies?',
 'What led the author to contemplate a career in art instead of computer science?']
```



```python
hypothetical_questions = chain.batch(docs, {"max_concurrency": 5})
```



```python
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="hypo-questions", embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]
```



```python
question_docs = []
for i, question_list in enumerate(hypothetical_questions):
    question_docs.extend(
        [Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list]
    )
```



```python
retriever.vectorstore.add_documents(question_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
```



```python
sub_docs = vectorstore.similarity_search("justice breyer")
```



```python
sub_docs
```



```text
[Document(page_content='Who has been nominated to serve on the United States Supreme Court?', metadata={'doc_id': '0b3a349e-c936-4e77-9c40-0a39fc3e07f0'}),
 Document(page_content="What was the context and content of Robert Morris' advice to the document's author in 2010?", metadata={'doc_id': 'b2b2cdca-988a-4af1-ba47-46170770bc8c'}),
 Document(page_content='How did personal circumstances influence the decision to pass on the leadership of Y Combinator?', metadata={'doc_id': 'b2b2cdca-988a-4af1-ba47-46170770bc8c'}),
 Document(page_content='What were the reasons for the author leaving Yahoo in the summer of 1999?', metadata={'doc_id': 'ce4f4981-ca60-4f56-86f0-89466de62325'})]
```



```python
retrieved_docs = retriever.get_relevant_documents("justice breyer")
```



```python
len(retrieved_docs[0].page_content)
```



```text
9194
```



[
  ](https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder)