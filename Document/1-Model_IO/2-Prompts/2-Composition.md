# 组合

LangChain 提供了一个用户友好的界面，用于将提示的各个部分组合在一起。您可以使用字符串提示或聊天提示来执行此操作。通过这种方式构建提示，可以轻松重用组件。

## 字符串提示组合

在处理字符串提示时，每个模板将连接在一起。您可以直接处理提示或字符串（列表中的第一个元素必须是`prompt`）

```python
from langchain.prompts import PromptTemplate
```

```python
prompt = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + "\n\nand in {language}"
)
```

```python
prompt
```

```text
PromptTemplate(input_variables=['language', 'topic'], output_parser=None, partial_variables={}, template='Tell me a joke about {topic}, make it funny\n\nand in {language}', template_format='f-string', validate_template=True)
```

```python
prompt.format(topic="sports", language="spanish")
```

```text
'Tell me a joke about sports, make it funny\n\nand in spanish'
```

You can also use it in an LLMChain, just like before.

```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
```

```python
openai_api_key = "EMPTY"
openai_api_base = "http://10.10.11.2:8000/v1"

model = ChatOpenAI( 
    openai_api_key=openai_api_key, 
    openai_api_base=openai_api_base,
    model="Qwen-72B-Chat",
)
```

```python
chain = LLMChain(llm=model, prompt=prompt)
```

```python
chain.run(topic="sports", language="spanish")
```

```text
'¿Por qué el futbolista llevaba un paraguas al partido?\n\nPorque pronosticaban lluvia de goles.'
```

## 聊天提示符的组成

聊天提示符由一系列消息组成。纯粹是为了开发者的体验，我们添加了一种方便的方式来创建这些提示符。在这个管道中，每个新元素都是最终提示符中的新消息。
```python
from langchain.schema import AIMessage, HumanMessage, SystemMessage
```

首先，让我们使用一个系统消息初始化基础`ChatPromptTemplate`。它不必以系统消息开始，但通常是一个好习惯。
```python
prompt = SystemMessage(content="You are a nice pirate")
```

然后，你可以轻松地创建一个管道，将它与其他消息或消息模板结合在一起。当没有变量要格式化时，使用`Message`；当有变量要格式化时，使用`MessageTemplate`。你也可以只使用一个字符串（注意：这将自动被推断为`HumanMessagePromptTemplate`）。

```python
new_prompt = (
    prompt + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"
)
```

在引擎下，这会创建一个`ChatPromptTemplate`类的实例，所以你可以像以前一样使用它！

```python
new_prompt.format_messages(input="i said hi")
```

```text
[SystemMessage(content='You are a nice pirate', additional_kwargs={}),
 HumanMessage(content='hi', additional_kwargs={}, example=False),
 AIMessage(content='what?', additional_kwargs={}, example=False),
 HumanMessage(content='i said hi', additional_kwargs={}, example=False)]
```

你也可以像以前一样，在`LLMChain`中使用它。

```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
```

```python
model = ChatOpenAI()
```

```python
chain = LLMChain(llm=model, prompt=new_prompt)
```

```python
chain.run("i said hi")
```

```text
'Oh, hello! How can I assist you today?'
```