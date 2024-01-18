# Contextual compression 情境压缩

一种挑战是，通常在将数据输入系统时，您不知道文档存储系统将面临的特定查询。这意味着与查询最相关的信息可能被包含大量不相关文本的文档所掩盖。将整个文档通过您的应用程序传递可能会导致更昂贵的LLM调用和较差的响应。

情境压缩旨在解决此问题。这个想法很简单：不是立即按原样返回检索到的文档，而是使用给定查询的上下文压缩它们，以便只返回相关的信息。这里的“压缩”指的是同时压缩单个文档的内容和过滤掉文档。

要使用情境压缩检索器，您需要：- 基本检索器 - 文档压缩器

情境压缩检索器将查询传递给基本检索器，获取初始文档并将它们通过文档压缩器。文档压缩器接受一个文档列表并将其缩短，方法是减少文档内容或完全删除文档。

![img](../../../../contextual_compression.jpg)

## Get started

```python
# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
```



## Using a vanilla vector store retriever

让我们首先初始化一个简单的向量存储检索器，并存储2023年国情咨文（分块存储）。我们可以看到，对于给定的一个示例问题，我们的检索器会返回一到两个相关的文档和几个不相关的文档。即使相关文档中也包含大量不相关的信息。

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

documents = TextLoader("../../state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()

docs = retriever.get_relevant_documents(
    "What did the president say about Ketanji Brown Jackson"
)
pretty_print_docs(docs)
```



```text
文档1：

今晚，我呼吁参议院：通过《自由投票法案》；通过《约翰·刘易斯投票权法案》；同时，通过《披露法案》，让美国人知道谁在资助我们的选举。

今晚，我想向一位为国家服务一生的人致敬：斯蒂芬·布雷耶法官——一名陆军退伍军人、宪法学者和即将退休的美国最高法院大法官。布雷耶法官，感谢您的服务。

总统的最严重的宪法责任之一是提名某人担任美国最高法院法官。

我在4天前就做到了，当时我提名了巡回上诉法院法官凯坦吉·布朗·杰克逊。她是我国最优秀的法律人才之一，将继续布雷耶法官的卓越传统。

文档2：

曾任私人执业的顶级诉讼律师。曾任联邦公设辩护人。来自一个由公立学校教育工作者和警察组成的家庭。是一名共识建设者。自从她被提名以来，她得到了广泛的支持——从警察兄弟会到民主党人和共和党人任命的前法官。

如果我们想要推进自由和正义，我们需要保障边境安全并修复移民系统。

我们可以做到这两点。在边境，我们已经安装了新技术，如先进的扫描仪，以更好地检测毒品走私。

我们与墨西哥和危地马拉设立了联合巡逻队，以抓获更多的人口贩运者。

我们正在建立专门的移民法官，以便因受迫害和暴力而逃离的家庭能够更快地审理案件。

我们正在争取承诺并支持南美和中美洲的合作伙伴，以接纳更多难民并保障他们自己的边境安全。

文档3：

对于我们的LGBTQ+美国人，让我们最终将《两党平等法案》送到我的办公桌上。针对跨性别美国人及其家庭的州法律泛滥是错误的。

正如我在去年所说，尤其是对我们年轻的跨性别美国人，作为总统，我将始终支持你，让你做自己，发挥上天赋予你的潜力。

虽然似乎我们从未达成一致，但事实并非如此。去年我签署了80项两党法案成为法律。从防止政府停摆，到保护亚裔美国人免受仍太常见的仇恨犯罪，到改革军事司法。

很快，我们将加强我30年前首次撰写的《反对暴力侵害妇女法》。对我们来说，展现出我们能够团结一致并做大事是很重要的。

因此，今晚我将提出国家团结议程。我们可以一起做的四件大事。

首先，战胜阿片类药物成瘾。

文档4：

今晚，我将宣布打击这些公司向美国企业和消费者收取过高费用的行动。

随着华尔街公司接管更多的养老院，这些养老院的质量下降，成本上升。

在我的任期内，这种现象将结束。

医疗保险将为养老院设定更高的标准，确保您的亲人得到应有的照顾。

我们还将通过提高工人的待遇、提供更多的培训和学徒制、根据技能而非学位雇用来降低成本并保持经济强劲发展。

让我们通过《工资公平法》和带薪休假。

将最低工资提高到每小时15美元，并延长儿童税收抵免，以便没有人必须在贫困中抚养家庭。

让我们增加佩尔奖学金，并增加对我们历史上对HBCUs的支持，并投资于我们第一夫人吉尔（她全职教书）所说的美国最好的秘密：社区学院。
```



## Adding contextual compression with an `LLMChainExtractor`

现在让我们用`ContextualCompressionRetriever`来包装我们的基础检索器。我们将添加一个`LLMChainExtractor`，它将遍历最初返回的文档，并从每个文档中提取与查询相关的仅内容。

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)
```



```text
/Users/harrisonchase/workplace/langchain/libs/langchain/langchain/chains/llm.py:316: UserWarning: The predict_and_parse method is deprecated, instead pass an output parser directly to LLMChain.
  warnings.warn(
/Users/harrisonchase/workplace/langchain/libs/langchain/langchain/chains/llm.py:316: UserWarning: The predict_and_parse method is deprecated, instead pass an output parser directly to LLMChain.
  warnings.warn(
/Users/harrisonchase/workplace/langchain/libs/langchain/langchain/chains/llm.py:316: UserWarning: The predict_and_parse method is deprecated, instead pass an output parser directly to LLMChain.
  warnings.warn(
/Users/harrisonchase/workplace/langchain/libs/langchain/langchain/chains/llm.py:316: UserWarning: The predict_and_parse method is deprecated, instead pass an output parser directly to LLMChain.
  warnings.warn(
```



```text
Document 1:

I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson.
```



## More built-in compressors: filters

### `LLMChainFilter`

`LLMChainFilter`是一个略微简单但更健壮的压缩器，它使用LLM链来决定从初始检索出的文档中过滤掉哪些，返回哪些，而不操作文档内容。

```python
from langchain.retrievers.document_compressors import LLMChainFilter

_filter = LLMChainFilter.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=retriever
)

compressed_docs = compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)
```



```text
/Users/harrisonchase/workplace/langchain/libs/langchain/langchain/chains/llm.py:316: UserWarning: The predict_and_parse method is deprecated, instead pass an output parser directly to LLMChain.
  warnings.warn(
/Users/harrisonchase/workplace/langchain/libs/langchain/langchain/chains/llm.py:316: UserWarning: The predict_and_parse method is deprecated, instead pass an output parser directly to LLMChain.
  warnings.warn(
/Users/harrisonchase/workplace/langchain/libs/langchain/langchain/chains/llm.py:316: UserWarning: The predict_and_parse method is deprecated, instead pass an output parser directly to LLMChain.
  warnings.warn(
/Users/harrisonchase/workplace/langchain/libs/langchain/langchain/chains/llm.py:316: UserWarning: The predict_and_parse method is deprecated, instead pass an output parser directly to LLMChain.
  warnings.warn(
```



```text
Document 1:

Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. 

Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. 

One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. 

And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
```



### `EmbeddingsFilter`

添加针对每个检索到的文档的额外LLM调用是昂贵且耗时的。`EmbeddingsFilter`通过将文档和查询嵌入其中，并仅返回与查询具有足够相似嵌入的文档，提供了一个更便宜、更快的选择。

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=retriever
)

compressed_docs = compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)
```



```text
Document 1:

Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. 

Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. 

One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. 

And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
----------------------------------------------------------------------------------------------------
Document 2:

A former top litigator in private practice. A former federal public defender. And from a family of public school educators and police officers. A consensus builder. Since she’s been nominated, she’s received a broad range of support—from the Fraternal Order of Police to former judges appointed by Democrats and Republicans. 

And if we are to advance liberty and justice, we need to secure the Border and fix the immigration system. 

We can do both. At our border, we’ve installed new technology like cutting-edge scanners to better detect drug smuggling.  

We’ve set up joint patrols with Mexico and Guatemala to catch more human traffickers.  

We’re putting in place dedicated immigration judges so families fleeing persecution and violence can have their cases heard faster. 

We’re securing commitments and supporting partners in South and Central America to host more refugees and secure their own borders.
----------------------------------------------------------------------------------------------------
Document 3:

And for our LGBTQ+ Americans, let’s finally get the bipartisan Equality Act to my desk. The onslaught of state laws targeting transgender Americans and their families is wrong. 

As I said last year, especially to our younger transgender Americans, I will always have your back as your President, so you can be yourself and reach your God-given potential. 

While it often appears that we never agree, that isn’t true. I signed 80 bipartisan bills into law last year. From preventing government shutdowns to protecting Asian-Americans from still-too-common hate crimes to reforming military justice. 

And soon, we’ll strengthen the Violence Against Women Act that I first wrote three decades ago. It is important for us to show the nation that we can come together and do big things. 

So tonight I’m offering a Unity Agenda for the Nation. Four big things we can do together.  

First, beat the opioid epidemic.
```



## Stringing compressors and document transformers together

使用`DocumentCompressorPipeline`，我们也可以轻松地按顺序组合多个压缩器。除了压缩器，我们还可以在管道中添加`BaseDocumentTransformer`，这些转换器不会执行任何上下文压缩，而只是对一组文档执行一些转换。例如，可以使用`TextSplitter`作为文档转换器将文档拆分为较小的片段，而`EmbeddingsRedundantFilter`可以用于根据文档之间的嵌入相似性过滤掉冗余文档。

以下是我们首先将文档拆分为较小的片段，然后删除冗余文档，然后根据查询的相关性进行过滤的压缩器管道的创建方法。

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)
```



```python
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)
```



```text
Document 1:

One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. 

And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson
----------------------------------------------------------------------------------------------------
Document 2:

As I said last year, especially to our younger transgender Americans, I will always have your back as your President, so you can be yourself and reach your God-given potential. 

While it often appears that we never agree, that isn’t true. I signed 80 bipartisan bills into law last year
----------------------------------------------------------------------------------------------------
Document 3:

A former top litigator in private practice. A former federal public defender. And from a family of public school educators and police officers. A consensus builder
----------------------------------------------------------------------------------------------------
Document 4:

Since she’s been nominated, she’s received a broad range of support—from the Fraternal Order of Police to former judges appointed by Democrats and Republicans. 

And if we are to advance liberty and justice, we need to secure the Border and fix the immigration system. 

We can do both
```
