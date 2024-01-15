# MultiQueryRetriever多查询检索

基于距离的向量数据库检索将查询嵌入到高维空间中，并根据“距离”找到相似的嵌入文档。但是，检索可能会因为查询措辞的微妙变化或嵌入未能捕捉到数据的语义而产生不同的结果。有时会进行提示工程/调节来手动解决这些问题，但可能会很烦琐。

`MultiQueryRetriever`通过使用LLM为给定的用户输入查询生成多个查询来自动化提示调节的过程。对于每个查询，它都会检索一组相关的文档，并取所有查询的唯一交集以获取更大可能相关的文档集。通过从同一问题生成多个视角，`MultiQueryRetriever`可能会克服基于距离的检索的一些局限性，并获取更丰富的结果。

```python
# Build a sample vectorDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load blog post
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# VectorDB
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
```

#### Simple usage

指定用于查询生成的LLM，检索器将完成其余工作。

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

question = "What are the approaches to Task Decomposition?"
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)
```



```python
# Set logging for the queries
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
```



```python
unique_docs = retriever_from_llm.get_relevant_documents(query=question)
len(unique_docs)
```



```text
INFO:langchain.retrievers.multi_query:Generated queries: ['1. How can Task Decomposition be approached?', '2. What are the different methods for Task Decomposition?', '3. What are the various approaches to decomposing tasks?']
```



```text
5
```



#### 提供自己的提示

您也可以提供一个提示，以及一个输出解析器来将结果分割成一个查询列表。

```python
from typing import List

from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field


# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
llm = ChatOpenAI(temperature=0)

# Chain
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)

# Other inputs
question = "What are the approaches to Task Decomposition?"
```



```python
# Run
retriever = MultiQueryRetriever(
    retriever=vectordb.as_retriever(), llm_chain=llm_chain, parser_key="lines"
)  # "lines" is the key (attribute name) of the parsed output

# Results
unique_docs = retriever.get_relevant_documents(
    query="What does the course say about regression?"
)
len(unique_docs)
```



```text
INFO:langchain.retrievers.multi_query:Generated queries: ["1. What is the course's perspective on regression?", '2. Can you provide information on regression as discussed in the course?', '3. How does the course cover the topic of regression?', "4. What are the course's teachings on regression?", '5. In relation to the course, what is mentioned about regression?']
```



```text
11
```