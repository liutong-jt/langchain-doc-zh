# Self-querying

前往[Integrations](https://python.langchain.com/docs/integrations/retrievers/self_query)获取关于内置自我查询支持的向量存储的文档。

正如名称所示，自我查询检索器是一种具有查询自身能力的检索器。具体来说，给定任何自然语言查询，检索器使用查询构建LLM链来编写结构化查询，然后将该结构化查询应用于其底层的VectorStore。这使得检索器不仅可以使用用户输入的查询与存储文档的内容进行语义相似性比较，还可以从用户查询中提取存储文档元数据的过滤器，并执行这些过滤器。

![img](https://drive.google.com/uc?id=1OQUN-0MJcDUxmPXofgS7MqReEs720pqS.png)

## Get started

为了演示目的，我们将使用`Chroma`向量存储。我们创建了一小批文档演示集，其中包含电影摘要。 

**注意：** 自我查询检索器需要您安装`lark`包。

```python
%pip install --upgrade --quiet  lark chromadb
```

```python
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

docs = [
    Document(
        page_content="一群科学家带回了恐龙，然后混乱爆发了",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="莱昂纳多·迪卡普里奥在一个又一个梦境中迷失了自我……",
        metadata={"year": 2010, "director": "克里斯托弗·诺兰", "rating": 8.2},
    ),
    Document(
        page_content="一个心理学家/侦探在一个又一个的梦中迷失了方向，而《盗梦空间》重新使用了这个想法。",
        metadata={"year": 2006, "director": "宫崎骏", "rating": 8.6},
    ),
    Document(
        page_content="一群正常体型的女性非常纯真，有些男性会对她们念念不忘。",
        metadata={"year": 2019, "director": "格蕾塔·葛韦格", "rating": 8.3},
    ),
    Document(
        page_content="玩具变得活灵活现，玩得不亦乐乎。",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="三名男子走进区域，三名男子走出区域。",
        metadata={
            "year": 1979,
            "director": "安德烈·塔可夫斯基",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
```

### Creating our self-querying retriever

现在我们可以实例化我们的检索器。为此，我们需要提前提供一些关于我们的文档支持的元数据字段和文档内容的简短描述的信息。

```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

metadata_field_info = [ 
    AttributeInfo( name="genre", 
                  description="电影的类型。可以是 ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated'] 中的一个", 
                  type="string", ), 
    AttributeInfo( name="year", 
                  description="电影发行的年份", 
                  type="integer", ), 
    AttributeInfo( name="director", 
                  description="电影导演的名字", 
                  type="string", ), 
    AttributeInfo( name="rating", 
                  description="电影的1-10评分", 
                  type="float" ), 
]

document_content_description = "Brief summary of a movie"
llm = ChatOpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)
```

### Testing it out

And now we can actually try using our retriever!

```python
# This example only specifies a filter
retriever.invoke("I want to watch a movie rated higher than 8.5")
```

```text
[Document(page_content='三名男子走进区域，三名男子走出区域。', metadata={'director': 'Andrei Tarkovsky', 'genre': 'thriller', 'rating': 9.9, 'year': 1979}),
 Document(page_content='A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea', metadata={'director': 'Satoshi Kon', 'rating': 8.6, 'year': 2006})]
```

```python
# This example specifies a query and a filter
retriever.invoke("格蕾塔·葛韦格有没有执导过关于女性的电影")
```

```text
[Document(page_content='一群正常体型的女性非常纯真，有些男性会对她们念念不忘。', metadata={'director': 'Greta Gerwig', 'rating': 8.3, 'year': 2019})]
```

```python
# This example specifies a composite filter
retriever.invoke("What's a highly rated (above 8.5) science fiction film?")
```

```text
[Document(page_content='一个心理学家/侦探在一个又一个的梦中迷失了方向，而《盗梦空间》重新使用了这个想法。', metadata={'director': 'Satoshi Kon', 'rating': 8.6, 'year': 2006}),
 Document(page_content='Three men walk into the Zone, three men walk out of the Zone', metadata={'director': 'Andrei Tarkovsky', 'genre': 'thriller', 'rating': 9.9, 'year': 1979})]
```

```python
# This example specifies a query and composite filter
retriever.invoke(
    "请问有没有一部1990年以后、2005年以前上映的关于玩具的电影，最好是动画片？"
)
```

```text
[Document(page_content='Toys come alive and have a blast doing so', metadata={'genre': 'animated', 'year': 1995})]
```

### Filter k

我们也可以使用self查询检索器来指定`k`：要获取的文档数量。

我们可以通过在构造函数中传递`enable_limit=True`来实现这一点。

```python
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
)

# This example only specifies a relevant query
retriever.invoke("What are two movies about dinosaurs")
```



```text
[Document(page_content='A bunch of scientists bring back dinosaurs and mayhem breaks loose', metadata={'genre': 'science fiction', 'rating': 7.7, 'year': 1993}),
 Document(page_content='Toys come alive and have a blast doing so', metadata={'genre': 'animated', 'year': 1995})]
```



## Constructing from scratch with LCEL

为了了解检索器的工作原理，并且有更多的自定义控制，我们可以从头开始重新构建检索器。

首先，我们需要创建一个查询构造链。这个链将接受用户查询，并生成一个`StructuredQuery`对象，该对象捕获了用户指定的过滤器。我们提供了一些用于创建提示和输出解析器的辅助函数。这些函数有许多可调参数，但我们在这里为了简单起见将忽略它们。

```python
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
)
output_parser = StructuredQueryOutputParser.from_components()
query_constructor = prompt | llm | output_parser
```



Let’s look at our prompt:

```python
print(prompt.format(query="dummy question"))
```



~~~text
Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{
    "query": string \ text string to compare to document contents
    "filter": string \ logical condition statement for filtering documents
}
```

The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` (eq | ne | gt | gte | lt | lte | contain | like | in | nin): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` (and | or | not): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters only use the attributed names with its function names if there are functions applied on them.
Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.

<< Example 1. >>
Data Source:
```json
{
    "content": "Lyrics of a song",
    "attributes": {
        "artist": {
            "type": "string",
            "description": "Name of the song artist"
        },
        "length": {
            "type": "integer",
            "description": "Length of the song in seconds"
        },
        "genre": {
            "type": "string",
            "description": "The song genre, one of "pop", "rock" or "rap""
        }
    }
}
```

User Query:
What are songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre

Structured Request:
```json
{
    "query": "teenager love",
    "filter": "and(or(eq(\"artist\", \"Taylor Swift\"), eq(\"artist\", \"Katy Perry\")), lt(\"length\", 180), eq(\"genre\", \"pop\"))"
}
```


<< Example 2. >>
Data Source:
```json
{
    "content": "Lyrics of a song",
    "attributes": {
        "artist": {
            "type": "string",
            "description": "Name of the song artist"
        },
        "length": {
            "type": "integer",
            "description": "Length of the song in seconds"
        },
        "genre": {
            "type": "string",
            "description": "The song genre, one of "pop", "rock" or "rap""
        }
    }
}
```

User Query:
What are songs that were not published on Spotify

Structured Request:
```json
{
    "query": "",
    "filter": "NO_FILTER"
}
```


<< Example 3. >>
Data Source:
```json
{
    "content": "Brief summary of a movie",
    "attributes": {
    "genre": {
        "description": "The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        "type": "string"
    },
    "year": {
        "description": "The year the movie was released",
        "type": "integer"
    },
    "director": {
        "description": "The name of the movie director",
        "type": "string"
    },
    "rating": {
        "description": "A 1-10 rating for the movie",
        "type": "float"
    }
}
}
```

User Query:
dummy question

Structured Request:
~~~



And what our full chain produces:

```python
query_constructor.invoke(
    {
        "query": "What are some sci-fi movies from the 90's directed by Luc Besson about taxi drivers"
    }
)
```



```text
StructuredQuery(query='taxi driver', filter=Operation(operator=<Operator.AND: 'and'>, arguments=[Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='genre', value='science fiction'), Operation(operator=<Operator.AND: 'and'>, arguments=[Comparison(comparator=<Comparator.GTE: 'gte'>, attribute='year', value=1990), Comparison(comparator=<Comparator.LT: 'lt'>, attribute='year', value=2000)]), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='director', value='Luc Besson')]), limit=None)
```

查询构造器是自我查询检索器的关键元素。要构建一个好的检索系统，您需要确保您的查询构造器运行良好。这通常需要调整提示、提示中的示例、属性描述等。要查看有关如何在一些酒店库存数据上细化查询构造器的示例，请[查看此食谱](https://github.com/langchain-ai/langchain/blob/master/cookbook/self_query_hotel_search.ipynb)。

下一个关键元素是结构化查询翻译器。这是负责将通用`StructuredQuery`对象转换为您正在使用的向量存储的元数据过滤器的语法的对象。LangChain附带了许多内置翻译器。要查看它们，请转到[Integrations部分](https://python.langchain.com/docs/integrations/retrievers/self_query)。

```python
from langchain.retrievers.self_query.chroma import ChromaTranslator

retriever = SelfQueryRetriever(
    query_constructor=query_constructor,
    vectorstore=vectorstore,
    structured_query_translator=ChromaTranslator(),
)
```



```python
retriever.invoke(
    "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
)
```



```text
[Document(page_content='Toys come alive and have a blast doing so', metadata={'genre': 'animated', 'year': 1995})]
```



[
  ](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)