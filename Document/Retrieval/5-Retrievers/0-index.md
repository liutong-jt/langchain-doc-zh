# 检索器
检索器是一个接口，它根据未结构化的查询返回文档。它比向量存储更为通用。检索器不必能够存储文档，只需返回（或检索）文档即可。向量存储可以作为检索器的后端，但还有其他类型的检索器。

检索器接受一个字符串查询作为输入，返回一个`Document`列表作为输出。

## 先进的检索类型

LangChain提供了几种先进的检索类型。下面是完整的列表，以及以下信息：

**名称**：检索算法的名称。

**索引类型**：此方法依赖的索引类型（如果有）。

**使用 LLM**：此检索方法是否使用 LLM。

**使用场景**：关于我们何时应该考虑使用此检索方法的评论。

**描述**：描述此检索算法正在做什么。

| 名称                                                         | 索引类型          | 使用LLM    | 何时使用                                                     | 描述                                                         |      |
| ------------------------------------------------------------ | ----------------- | ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| [Vectorstore](https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore) | 向量存储          | 否         | 如果您刚刚开始并且正在寻找简单快捷的方法                     | 这是最简单的方法，也是最容易开始的方法。它涉及为每个文本创建嵌入。 |      |
| [ParentDocument](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever) | 向量存储+文档存储 | 否         | 如果您的页面包含许多较小的、彼此独立的信息，但最好一起检索   | 这涉及到为每个文档索引多个块。然后，您找到嵌入空间中最相似的块，但您检索整个父文档并返回它（而不是单独的块）。 |      |
| [Multi Vector](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector) | 向量存储+文档存储 | 有时在索引 | 如果您能够从文档中提取出比文本本身更有用的信息               | 这涉及为每个文档创建多个向量。每个向量可以以许多不同的方式创建-示例包括文本的摘要和假设问题。 |      |
| [Self Query](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query) | 向量存储          | 是         | 如果用户正在提问，这些问题是通过基于元数据而不是文本相似性的文档检索来更好地回答的 | 这使用LLM将用户输入转换为两件事：（1）要查找语义的字符串，（2）与之配套的元数据过滤器。这很有用，因为通常情况下，问题都是关于文档的元数据（而不是内容本身）。 |      |
| [Contextual Compression](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression) | 任何              | 有时       | 如果您发现检索的文档包含太多不相关的信息，它们分散了LLM的注意力 | 这在另一个检索器的顶部添加了一个后处理步骤，提取检索文档中只包含最相关信息的部分。这可以使用嵌入或LLM完成。 |      |
| [Time-Weighted Vectorstore](https://python.langchain.com/docs/modules/data_connection/retrievers/time_weighted_vectorstore) | 向量存储          | 否         | 如果您有关于您的文档的日期和时间戳，且您想检索最近的文档     | 这根据语义相似性（如通常的向量检索）和时间紧迫性（查看索引文档的日期和时间戳）来检索文档。 |      |
| [Multi-Query Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever) | 任何              | 是         | 如果用户正在提问，这些问题是复杂的，并需要多个独立的信息来回答 | 这使用LLM从原始查询生成多个查询。这很有用，因为原始查询需要关于多个主题的多个信息才能正确回答。通过生成多个查询，我们可以在其中为每个查询检索文档。 |      |
| [Ensemble](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble) | 任何              | 否         | 如果您有多个检索方法，并想尝试将它们组合在一起               | 这从多个检索器检索文档，然后将它们组合在一起。               |      |
| [Long-Context Reorder](https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder) | 任何              | 否         | 如果您正在使用一个长上下文模型，并注意到它没有注意检索文档中中间部分的信息 | 这从底层检索器检索文档，然后重新排列这些文档，以便最相似的文档位于开始和结束附近。这很有用，因为已经证明，对于较长上下文模型，它们有时不会注意检索文档中中间部分的信息。 |      |

## 第三方集成

LangChain还与许多第三方检索服务集成。要获取完整列表，请参阅[此列表](https://python.langchain.com/docs/integrations/retrievers/)以获取所有集成。

## 在LCEL中使用检索器

由于检索器是Runnable的，因此我们可以轻松地将它们与其他Runnable对象组合：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke("What did the president say about technology?")
```



## 定制检索器

因为检索器接口非常简单，所以编写一个定制的检索器非常容易。

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List


class CustomRetriever(BaseRetriever):
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [Document(page_content=query)]

retriever = CustomRetriever()

retriever.get_relevant_documents("bar")
```