# 快速入门

提示模板是为语言模型生成提示的预定义配方。

模板可以包括指令、少量示例以及特定的任务相关的上下文和问题。

LangChain提供了用于创建和使用提示模板的工具。

LangChain致力于创建与模型无关的模板，以便在不同的语言模型之间轻松重用现有模板。

通常，语言模型期望提示要么是一个字符串，要么是一个聊天消息的列表。

## **`PromptTemplate`**

使用`PromptTemplate`创建字符串提示的模板。

默认情况下，`PromptTemplate`使用Python的[str.format](https://docs.python.org/3/library/stdtypes.html#str.format)语法 进行模板化。

```python
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
prompt_template.format(adjective="funny", content="chickens")
```

```text
'Tell me a funny joke about chickens.'
```

The template supports any number of variables, including no variables:

```python
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke")
prompt_template.format()
```

```text
'Tell me a joke'
```

您可以创建自定义提示模板，以您想要的任何方式格式化提示。 有关更多信息，请参阅[自定义提示模板](./custom_prompt_template)。

## `ChatPromptTemplate`

聊天模型的提示是一个聊天消息列表。

每个聊天消息都与内容关联，并且还有一个称为 `role` 的附加参数。 例如，在 OpenAI [Chat Completions API](https://platform.openai.com/docs/guides/chat/introduction) 中，聊天消息可以与 AI 助手、人类或系统角色关联。

按如下方式创建聊天提示模板：

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(name="Bob", user_input="What is your name?")
```

`ChatPromptTemplate.from_messages` 接受各种消息表示形式。

例如，除了使用上面使用的 (type, content) 2 元组表示形式外，您还可以传递 `MessagePromptTemplate` 或 `BaseMessage` 的实例。

```python
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more upbeat."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)
messages = chat_template.format_messages(text="I don't like eating tasty things")
print(messages)
```

```text
[SystemMessage(content="You are a helpful assistant that re-writes the user's text to sound more upbeat."), HumanMessage(content="I don't like eating tasty things")]
```

This provides you with a lot of flexibility in how you construct your chat prompts.

## LCEL

`PromptTemplate` 和 `ChatPromptTemplate` 实现了 [Runnable 接口](/docs/expression_language/interface)，这是 [LangChain 表达式语言 (LCEL)](/docs/expression_language/) 的基本构建块。这意味着它们支持 `invoke`、`ainvoke`、`stream`、`astream`、`batch`、`abatch`、`astream_log` 调用。

`PromptTemplate` 接受一个字典（提示变量）并返回一个 `StringPromptValue`。`ChatPromptTemplate` 接受一个字典并返回一个 `ChatPromptValue`。

```python
prompt_val = prompt_template.invoke({"adjective": "funny", "content": "chickens"})
prompt_val
```

```python
StringPromptValue(text='Tell me a joke')
```

```python
prompt_val.to_string()
```

```text
'Tell me a joke'
```

```python
prompt_val.to_messages()
```

```python
[HumanMessage(content='Tell me a joke')]
```

```python
chat_val = chat_template.invoke({"text": "i dont like eating tasty things."})
```

```python
chat_val.to_messages()
```

```python
[SystemMessage(content="You are a helpful assistant that re-writes the user's text to sound more upbeat."),
 HumanMessage(content='i dont like eating tasty things.')]
```

```python
chat_val.to_string()
```

```python
"System: You are a helpful assistant that re-writes the user's text to sound more upbeat.\nHuman: i dont like eating tasty things."
```

