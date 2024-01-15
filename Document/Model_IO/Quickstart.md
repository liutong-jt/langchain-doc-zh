# Quickstart

# 快速入门

快速入门将涵盖使用语言模型的基础知识。它将介绍两种不同类型的模型 - LLMs 和 ChatModels。然后，它将介绍如何使用 PromptTemplates 格式化这些模型的
输入，以及如何使用 Output Parsers 处理输出。有关这些主题的深入概念指南，请参阅 [此文档](https://python.langchain.com/docs/modules/model_io/concepts)。

## 模型

对于这个入门指南，我们将提供两种选择：使用 OpenAI（一种可通过 API 获得的流行模型）或使用本地开源模型。

- OpenAI
- 本地

首先，我们需要安装他们的 Python 包：

```shell
pip install openai
```

访问API需要API密钥，您可以通过创建帐户并转到此处获得API密钥：https://platform.openai.com/account/api-keys。一旦我们有了密钥，我们就可以通过运行以下命令将其设置为环境变量：

```shell
export OPENAI_API_KEY="..."
```



We can then initialize the model:

```python
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI

llm = OpenAI()
chat_model = ChatOpenAI()
```



如果您不想设置环境变量，您可以在初始化OpenAI LLM类时通过`openai_api_key`命名参数直接传递密钥：

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(openai_api_key="...")
```

`llm`和`chat_model`都是表示特定模型配置的对象。您可以使用`temperature`等参数初始化它们，并将它们传递。它们之间的主要区别在于它们的输入和输出模
式。LLM对象接受字符串作为输入和输出字符串。ChatModel对象接受消息列表作为输入并输出消息。有关此差异的深入概念说明，请参阅[此文档](https://python.langchain.com/docs/modules/model_io/concepts)。

当我们调用它时，可以看到LLM和ChatModel之间的差异。

```python
from langchain.schema import HumanMessage

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]

llm.invoke(text)
# >> Feetful of Fun

chat_model.invoke(messages)
# >> AIMessage(content="Socks O'Color")
```

LLM 返回一个字符串，而 ChatModel 返回一条消息。
## 提示模板
大多数 LLM 应用程序不会直接将用户输入传递给 LLM。通常，它们会将用户输入添加到一个更大的文本中，称为提示模板，该文本提供了有关手头特定任务的附加上下文。
在上一个示例中，我们传递给模型的文本包含生成公司名称的说明。对于我们的应用程序，如果用户只需提供公司/产品的描述而不必担心向模型提供说明，那就太好了。
提示模板正好帮助解决这个问题！它们将从用户输入到完全格式化的提示的所有逻辑捆绑在一起。这可以非常简单地开始 - 例如，生成上述字符串的提示可能只是：

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
prompt.format(product="colorful socks")
```



```python
What is a good name for a company that makes colorful socks?
```

然而，使用这些方法而非原始字符串格式的优点有很多。你可以“部分”变量 - 例如，你可以一次只格式化部分变量。你可以将它们组合在一起，轻松地将不同的模板组合成一个提示。有关这些功能的解释，请参阅[关于提示的章节](https://python.langchain.com/docs/modules/model_io/prompts)以获取更多详细信息。

`PromptTemplate`也可以用于生成消息列表。在这种情况下，提示不仅包含有关内容的信息，而且还包含每个消息（其角色、在列表中的位置等）。在这里，最常
见的事情是`ChatPromptTemplate`是一个`ChatMessageTemplates`列表。每个`ChatMessageTemplate`都包含有关如何格式化该`ChatMessage`的说明 - 其角色，以及其内容。让我们看看下面的例子：

```python
from langchain.prompts.chat import ChatPromptTemplate

template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
```



```pycon
[
    SystemMessage(content="You are a helpful assistant that translates English to French.", additional_kwargs={}),
    HumanMessage(content="I love programming.")
]
```



也可以通过其他方式构建ChatPromptTemplates - 请参阅[prompts](https://python.langchain.com/docs/modules/model_io/prompts)部分以获得更详细的细节。

## 输出解析器

`OutputParser`将语言模型的原始输出转换为可用于下游的格式。有几种主要的`OutputParser`，包括：

- 将 `LLM` 中的文本转换为结构化信息（如 JSON）
- 将 `ChatMessage` 转换为一个字符串
- 将除了消息之外的函数调用返回的额外信息（如 OpenAI 函数调用）转换为字符串。

有关此信息的完整信息，请参阅[输出解析器部分](https://python.langchain.com/docs/modules/model_io/output_parsers)。

在这个入门指南中，我们使用一个简单的解析逗号分隔值列表的解析器。

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
output_parser.parse("hi, bye")
# >> ['hi', 'bye']
```



## 使用LCEL组合

现在我们可以将所有这些组合成一个链条。这个链条将采用输入变量，将这些变量传递给提示模板以创建提示，将提示传递给语言模型，然后将输出通过（可选）
输出解析器传递。这是一种将模块化逻辑捆绑在一起的便捷方式。让我们看看它在行动！

```python
template = "Generate a list of 5 {text}.\n\n{format_instructions}"

chat_prompt = ChatPromptTemplate.from_template(template)
chat_prompt = chat_prompt.partial(format_instructions=output_parser.get_format_instructions())
chain = chat_prompt | chat_model | output_parser
chain.invoke({"text": "colors"})
# >> ['red', 'blue', 'green', 'yellow', 'orange']
```


请注意，我们使用`|`语法将这些组件连接在一起。此`|`语法由LangChain表达式语言（LCEL）提供支持，并依赖于所有这些对象实现的通用`Runnable`接口。要了解有关LCEL的更多信息，请阅读此处的文档：https://python.langchain.com/docs/expression_language。

## 结论

这就是有关提示，模型和输出解析器的入门内容！这只是要学习的内容的表面。要了解更多信息，请参阅：

- [概念指南](https://python.langchain.com/docs/modules/model_io/concepts)，了解此处介绍的概念。
- [提示部分](https://python.langchain.com/docs/modules/model_io/prompts），了解如何使用提示模板。
- [LLM部分](https://python.langchain.com/docs/modules/model_io/llms)，了解有关LLM接口的更多信息。
- [ChatModel部分](https://python.langchain.com/docs/modules/model_io/chat)，了解有关ChatModel接口的更多信息。
- [输出解析器部分](https://python.langchain.com/docs/modules/model_io/output_parsers)，了解有关不同类型的输出解析器的信息。