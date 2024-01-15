# Vector stores

> INFO
>
> 前往[集成](https://python.langchain.com/docs/integrations/vectorstores/)获取关于内置与第三方矢量存储集成的文档。

存储和搜索非结构化数据最常见的方法之一是将非结构化数据嵌入并存储生成的嵌入向量，然后在查询时将非结构化查询嵌入并检索与嵌入查询“最相似”的嵌入向量。向量存储负责为您存储嵌入数据并进行向量搜索。

![Diagram illustrating the process of vector stores: 1. Load source data, 2. Query vector store, 3. Retrieve &#39;most similar&#39; results.](https://python.langchain.com/assets/images/vector_stores-125d1675d58cfb46ce9054c9019fea72.jpg)

## 开始

这个教程展示了一些与矢量存储相关的基础功能。在使用矢量存储时，关键部分是创建要放入其中的矢量，通常是通过嵌入创建的。因此，强烈建议您在深入研究之前熟悉一下[文本嵌入模型](https://python.langchain.com/docs/modules/data_connection/text_embedding/)接口。

有许多优秀的矢量存储选项，以下是一些免费、开源且完全在本地机器上运行的选项。查看所有集成，获取许多优秀的托管服务。

- Chroma
- FAISS
- Lance

这个教程使用了运行在本地机器上的Chroma矢量数据库库。

```bash
pip install chromadb
```



我们想要使用OpenAIEmbeddings，所以我们需要获取OpenAI的API密钥。

```python
import os
import getpass

os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')
```



```python
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('../../../state_of_the_union.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OpenAIEmbeddings())
```



### Similarity search

```python
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)
```



```text
    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections.

    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.

    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.

    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
```



### Similarity search by vector

也可以使用`similarity_search_by_vector`来搜索与给定嵌入向量相似的文档，该方法接受一个嵌入向量作为参数，而不是一个字符串。

```python
embedding_vector = OpenAIEmbeddings().embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)
```



The query is the same, and so the result is also the same.

```text
    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections.

    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.

    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.

    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
```



## 异步操作

向量存储通常作为单独的服务运行，需要进行一些IO操作，因此可能被异步调用。这可以带来性能优势，因为你不必等待外部服务的响应。如果你在一个异步框架（如FastAPI）中工作，这可能也很重要。

LangChain支持向量存储的异步操作。所有方法都可能使用其异步对应方法，前缀为"a"，表示"异步"。

Qdrant是一个向量存储，支持所有异步操作，因此将在这个教程中使用。

```bash
pip install qdrant-client
```



```python
from langchain_community.vectorstores import Qdrant
```



### Create a vector store asynchronously

```python
db = await Qdrant.afrom_documents(documents, embeddings, "http://localhost:6333")
```



### Similarity search

```python
query = "What did the president say about Ketanji Brown Jackson"
docs = await db.asimilarity_search(query)
print(docs[0].page_content)
```



```text
    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections.

    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.

    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.

    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
```



### Similarity search by vector

```python
embedding_vector = embeddings.embed_query(query)
docs = await db.asimilarity_search_by_vector(embedding_vector)
```



## Maximum marginal relevance search (MMR)

最大化边际相关性优化了对查询的相似性和所选文档之间的多样性。它也受到异步API的支持。

```python
query = "What did the president say about Ketanji Brown Jackson"
found_docs = await qdrant.amax_marginal_relevance_search(query, k=2, fetch_k=10)
for i, doc in enumerate(found_docs):
    print(f"{i + 1}.", doc.page_content, "\n")
```



```text
1. Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections.

Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.

One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.

And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.

2. We can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together.

I recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera.

They were responding to a 9-1-1 call when a man shot and killed them with a stolen gun.

Officer Mora was 27 years old.

Officer Rivera was 22.

Both Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers.

I spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.

I’ve worked on these issues a long time.

I know what works: Investing in crime prevention and community police officers who’ll walk the beat, who’ll know the neighborhood, and who can restore trust and safety.
```