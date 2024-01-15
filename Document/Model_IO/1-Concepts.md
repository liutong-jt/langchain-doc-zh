## Concepts

任何语言模型应用程序的核心元素是...模型。LangChain为您提供与任何语言模型接口的构建块。本节中的所有内容都是关于使与模型一起工作变得更加容易。这主要涉及对模型的清晰接口，用于构建模型输入的辅助工具以及用于处理模型输出的辅助工具。

### Models

LangChain 集成的两种主要模型类型是 **LLMs** 和 **Chat** 模型。**它们由输入和输出类型定义**。

### LLMs

在LangChain中，LLMs 指的是纯文本完成模型。它们封装的API接受一个字符串提示作为输入，并输出一个字符串完成。OpenAI的GPT-3就是一个实现为LLM的模型。

### 聊天模型

聊天模型通常是通过LLMs提供支持，但已经针对对话进行了专门调整。关键在于，它们的提供商API使用与纯文本完成模型不同的接口。它们不是接受一个字符串，而是接受一个聊天消息列表作为输入，并返回一个AI消息作为输出。有关消息确切组成的更多信息，请参见下文。GPT-4和Anthropic的Claude-2都是实现为聊天模型的。

### Considerations 考虑因素

这两种API类型的输入和输出模式差别很大。这意味着与它们交互的最佳方式可能会有所不同。虽然LangChain使得可以将它们互换使用，但这并不意味着你**应该**这样做。特别是，LLMs和ChatModels的提示策略可能差别很大。这意味着你需要确保使用的提示是针对你正在使用的模型类型设计的。

### 消息

ChatModels 以消息列表作为输入并返回一条消息。有几种不同类型的消息。所有消息都有一个 `role` 和一个 `content` 属性。`role` 描述了谁在说这条
消息。LangChain 为不同的角色提供了不同的消息类。`content` 属性描述了消息的内容。这可以是几件不同的事情：

- 一个字符串（大多数模型都是这样的）
- 字典列表（这是用于多模态输入的，其中字典包含有关该输入类型和输入位置的信息）

此外，消息还有一个 `additional_kwargs` 属性。这是传递有关消息的附加信息的地方。这主要用于 *提供商特定* 的输入参数，而不是通用的。最知名的例子是 OpenAI 的 `function_call`。

#### HumanMessage

表示来自用户的消息。通常只包含内容。

#### AIMessage

这代表了模型发送的消息。它可能包含`additional_kwargs`，例如在使用OpenAI Function调用时的`functional_call`。

#### SystemMessage

这代表了一个系统消息。只有某些模型支持这个。这告诉模型如何表现。这通常只包含内容。

#### FunctionMessage

这表示函数调用的结果。除了`role`和`content`，此消息还有一个`name`参数，用于传达生成此结果时调用的函数的名称。

#### ToolMessage

这表示工具调用的结果。这与FunctionMessage不同，以匹配OpenAI的`function`和`tool`消息类型。除了`role`和`content`，此消息还有一个`tool_call_id`参数，用于传达生成此结果时调用工具的id。

### 提示

语言模型的输入通常称为提示。通常情况下，您的应用程序中的用户输入不是直接输入到模型中的。相反，他们的输入以某种方式转换为输入到模型中的字符串或消息列表。将用户输入转换为最终字符串或消息的对象称为“提示模板”。LangChain提供了几个抽象，以使处理提示变得更加容易。

#### PromptValue

ChatModels和LLMs接受不同的输入类型。PromptValue是一个设计为在两者之间互操作的类。它公开了一个方法，可以将其转换为字符串（以便与LLMs一起使
用）和另一个方法，可以将其转换为消息列表（以便与ChatModels一起使用）。

#### PromptTemplate

这是一个提示模板的例子。这由一个模板字符串组成。然后使用用户输入格式化此字符串以产生最终字符串。

#### MessagePromptTemplate

这是一个提示模板的例子。这由一个模板消息组成 - 意味着特定的角色和一个PromptTemplate。然后使用用户输入格式化此PromptTemplate，以产生最终字
符串，该字符串成为此消息的“内容”。

##### HumanMessagePromptTemplate

这是一个生成HumanMessage的MessagePromptTemplate。

##### AIMessagePromptTemplate

这是一个生成AIMessage的MessagePromptTemplate。

##### SystemMessagePromptTemplate

这是一个生成SystemMessage的MessagePromptTemplate。

#### MessagesPlaceholder

提示的输入通常是消息列表。这是您将使用MessagesPlaceholder的时候。这些对象由`variable_name`参数化。具有与该`variable_name`值相同的输入应该
是消息列表。

#### ChatPromptTemplate

这是一个提示模板的例子。这由一个MessagePromptTemplates或MessagePlaceholders列表组成。然后使用用户输入格式化这些内容，以产生最终的消息列表 。

### 输出解析器

模型的输出是字符串或消息。通常情况下，字符串或消息包含以特定格式格式化的下游（例如逗号分隔列表或JSON blob）中使用的信息。输出解析器负责接 收模型输出并将其转换为更可用的形式。这些通常在输出消息的`content`上工作，但偶尔也会在`additional_kwargs`字段中的值上工作。

#### StrOutputParser

这是一个简单的输出解析器，只需将语言模型（LLM或ChatModel）的输出转换为字符串即可。如果该模型是LLM（因此输出字符串），则只需通过该字符串。 如果输出是ChatModel（因此输出消息），则通过消息的`.content`属性通过。

#### OpenAI 函数解析器

有几个解析器专门用于处理OpenAI函数调用。它们接收`function_call`和`arguments`参数（在`additional_kwargs`中）的输出，并与这些参数一起工作， 很大程度上忽略了内容。

#### Agent Output Parsers 输出解析器

[代理]()是使用语言模型来确定要采取哪些步骤的系统。因此，语言模型的输出需要被解析成某种可以表示要采取哪些行动（如果有的话）的模式。AgentOutputParsers负责将原始LLM或ChatModel输出转换为该模式。这些输出解析器中的逻辑可以根据所使用的模型和提示策略而有所不同。

https://python.langchain.com/docs/modules/model_io/prompts/example_selector_types)

