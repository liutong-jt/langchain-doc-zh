# 输出解析器

语言模型输出文本。但很多时候，你可能想要获得比文本更结构化的信息。这就是输出解析器的作用。

输出解析器是一类帮助结构化语言模型响应的类。输出解析器必须实现两个主要方法：

- “获取格式说明”：一个返回字符串的方法，其中包含有关如何格式化语言模型输出的说明。
- “解析”：一个接受字符串（假设是语言模型的响应）并将其解析为某种结构的方法。

还有一个可选的：

- “解析带有提示”：一个接受字符串的方法（假设是语言模型的响应）和一个提示（假设是生成如此响应的提示）并将其解析为某种结构。提示主要提供给OutputParser，以便在某种程度上重新尝试或修复输出时使用提示中的信息。

## 开始使用

以下我们将介绍主要类型的输出解析器 `PydanticOutputParser`。

```python
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import OpenAI

model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)


# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# And a query intended to prompt a language model to populate the data structure.
prompt_and_model = prompt | model
output = prompt_and_model.invoke({"query": "Tell me a joke."})
parser.invoke(output)
```



```text
Joke(setup='Why did the chicken cross the road?', punchline='To get to the other side!')
```



## LCEL

输出解析器实现了[Runnable接口](https://python.langchain.com/docs/expression_language/interface)，这是[LangChain表达式语言（LCEL）](https://python.langchain.com/docs/expression_language/)的基本构建块。这意味着它们支持`invoke`、`ainvoke`、`stream`、`astream`、`batch`、`abatch`、`astream_log`调用。

输出解析器接受字符串或`BaseMessage`作为输入，可以返回任意类型。

```python
parser.invoke(output)
```



```text
Joke(setup='Why did the chicken cross the road?', punchline='To get to the other side!')
```



我们也可以直接将解析器添加到我们的`Runnable`序列中，而不是手动调用解析器：

```python
chain = prompt | model | parser
chain.invoke({"query": "Tell me a joke."})
```



```text
Joke(setup='Why did the chicken cross the road?', punchline='To get to the other side!')
```



虽然所有解析器都支持流式接口，但只有某些解析器可以在部分解析的对象中流式传输，因为这在很大程度上取决于输出类型。无法构造部分对象的解析器将简单地生成完全解析的输出。
例如，`SimpleJsonOutputParser`可以流式传输部分输出：

```python
from langchain.output_parsers.json import SimpleJsonOutputParser

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)
json_parser = SimpleJsonOutputParser()
json_chain = json_prompt | model | json_parser
```



```python
list(json_chain.stream({"question": "Who invented the microscope?"}))
```



```text
[{},
 {'answer': ''},
 {'answer': 'Ant'},
 {'answer': 'Anton'},
 {'answer': 'Antonie'},
 {'answer': 'Antonie van'},
 {'answer': 'Antonie van Lee'},
 {'answer': 'Antonie van Leeu'},
 {'answer': 'Antonie van Leeuwen'},
 {'answer': 'Antonie van Leeuwenho'},
 {'answer': 'Antonie van Leeuwenhoek'}]
```



虽然PydanticOutputParser不能

```python
list(chain.stream({"query": "Tell me a joke."}))
```



```text
[Joke(setup='Why did the chicken cross the road?', punchline='To get to the other side!')]
```