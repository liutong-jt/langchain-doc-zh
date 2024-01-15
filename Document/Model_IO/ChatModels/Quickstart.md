# 快速入门

聊天模型是语言模型的一种变体。虽然聊天模型在底层使用了语言模型，但它们使用的接口略有不同。它们不是使用“文本输入，文本输出”API，而是使用一种将“
聊天消息”作为输入和输出的接口。

## 设置

对于此示例，我们需要安装OpenAI Python包：

```bash
pip install openai
```



访问API需要API密钥，您可以通过创建帐户并转到此处获得API密钥：[https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)。一旦我们有了密钥，我们就可以通过运行以下命令将其设置为环境变量：

```bash
export OPENAI_API_KEY="..."
```

如果您不想设置环境变量，也可以在初始化OpenAI LLM类时通过`openai_api_key`命名参数直接传递密钥：

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(openai_api_key="...")
```



否则，您可以没有任何参数进行初始化：

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI()
```



## 消息

聊天模型接口是基于消息而不是原始文本的。LangChain 目前支持的消息类型有 `AIMessage`、`HumanMessage`、`SystemMessage`、`FunctionMessage` 和 `ChatMessage`，其中 `ChatMessage` 接受一个任意的角色参数。大多数情况下，您只需处理 `HumanMessage`、`AIMessage` 和 `SystemMessage`。

## LCEL

聊天模型实现了 [Runnable 接口](https://python.langchain.com/docs/expression_language/interface)，这是 [LangChain 表达式语言 (LCEL)](https://python.langchain.com/docs/expression_language/) 的基本构造块。这意味着它们支持 `invoke`、`ainvoke`、`stream`、`astream`、`batch`、`abatch`、`astream_log` 调用。

聊天模型接受 `List[BaseMessage]` 作为输入，或者可以强制转换为消息的对象，包括 `str`（转换为 `HumanMessage`）和 `PromptValue`。

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What is the purpose of model regularization?"),
]
```



```python
chat.invoke(messages)
```



```text
AIMessage(content="The purpose of model regularization is to prevent overfitting in machine learning models. Overfitting occurs when a model becomes too complex and starts to fit the noise in the training data, leading to poor generalization on unseen data. Regularization techniques introduce additional constraints or penalties to the model's objective function, discouraging it from becoming overly complex and promoting simpler and more generalizable models. Regularization helps to strike a balance between fitting the training data well and avoiding overfitting, leading to better performance on new, unseen data.")
```



```python
for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
```



```text
The purpose of model regularization is to prevent overfitting and improve the generalization of a machine learning model. Overfitting occurs when a model is too complex and learns the noise or random variations in the training data, which leads to poor performance on new, unseen data. Regularization techniques introduce additional constraints or penalties to the model's learning process, discouraging it from fitting the noise and reducing the complexity of the model. This helps to improve the model's ability to generalize well and make accurate predictions on unseen data.
```



```python
chat.batch([messages])
```



```text
[AIMessage(content="The purpose of model regularization is to prevent overfitting in machine learning models. Overfitting occurs when a model becomes too complex and starts to learn the noise or random fluctuations in the training data, rather than the underlying patterns or relationships. Regularization techniques add a penalty term to the model's objective function, which discourages the model from becoming too complex and helps it generalize better to new, unseen data. This improves the model's ability to make accurate predictions on new data by reducing the variance and increasing the model's overall performance.")]
```



```python
await chat.ainvoke(messages)
```



```text
AIMessage(content='The purpose of model regularization is to prevent overfitting in machine learning models. Overfitting occurs when a model becomes too complex and starts to memorize the training data instead of learning general patterns and relationships. This leads to poor performance on new, unseen data.\n\nRegularization techniques introduce additional constraints or penalties to the model during training, discouraging it from becoming overly complex. This helps to strike a balance between fitting the training data well and generalizing to new data. Regularization techniques can include adding a penalty term to the loss function, such as L1 or L2 regularization, or using techniques like dropout or early stopping. By regularizing the model, it encourages it to learn the most relevant features and reduce the impact of noise or outliers in the data.')
```



```python
async for chunk in chat.astream(messages):
    print(chunk.content, end="", flush=True)
```



```text
The purpose of model regularization is to prevent overfitting in machine learning models. Overfitting occurs when a model becomes too complex and starts to memorize the training data instead of learning the underlying patterns. Regularization techniques help in reducing the complexity of the model by adding a penalty to the loss function. This penalty encourages the model to have smaller weights or fewer features, making it more generalized and less prone to overfitting. The goal is to find the right balance between fitting the training data well and being able to generalize well to unseen data.
```



```python
async for chunk in chat.astream_log(messages):
    print(chunk)
```



```text
RunLogPatch({'op': 'replace',
  'path': '',
  'value': {'final_output': None,
            'id': '754c4143-2348-46c4-ad2b-3095913084c6',
            'logs': {},
            'streamed_output': []}})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='The')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' purpose')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' of')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' model')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' regularization')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' is')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' to')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' prevent')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' a')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' machine')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' learning')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' model')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' from')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' over')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='fit')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='ting')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' training')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' data')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' and')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' improve')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' its')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' general')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='ization')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' ability')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='.')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' Over')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='fit')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='ting')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' occurs')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' when')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' a')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' model')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' becomes')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' too')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' complex')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' and')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' learns')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' to')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' fit')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' noise')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' or')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' random')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' fluctuations')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' in')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' training')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' data')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=',')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' instead')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' of')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' capturing')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' underlying')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' patterns')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' and')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' relationships')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='.')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' Regular')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='ization')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' techniques')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' introduce')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' a')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' penalty')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' term')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' to')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' model')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content="'s")})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' objective')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' function')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=',')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' which')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' discour')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='ages')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' model')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' from')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' becoming')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' too')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' complex')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='.')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' This')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' helps')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' to')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' control')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' model')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content="'s")})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' complexity')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' and')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' reduces')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' risk')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' of')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' over')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='fit')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='ting')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=',')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' leading')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' to')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' better')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' performance')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' on')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' unseen')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' data')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='.')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='')})
RunLogPatch({'op': 'replace',
  'path': '/final_output',
  'value': {'generations': [[{'generation_info': {'finish_reason': 'stop'},
                              'message': AIMessageChunk(content="The purpose of model regularization is to prevent a machine learning model from overfitting the training data and improve its generalization ability. Overfitting occurs when a model becomes too complex and learns to fit the noise or random fluctuations in the training data, instead of capturing the underlying patterns and relationships. Regularization techniques introduce a penalty term to the model's objective function, which discourages the model from becoming too complex. This helps to control the model's complexity and reduces the risk of overfitting, leading to better performance on unseen data."),
                              'text': 'The purpose of model regularization is '
                                      'to prevent a machine learning model '
                                      'from overfitting the training data and '
                                      'improve its generalization ability. '
                                      'Overfitting occurs when a model becomes '
                                      'too complex and learns to fit the noise '
                                      'or random fluctuations in the training '
                                      'data, instead of capturing the '
                                      'underlying patterns and relationships. '
                                      'Regularization techniques introduce a '
                                      "penalty term to the model's objective "
                                      'function, which discourages the model '
                                      'from becoming too complex. This helps '
                                      "to control the model's complexity and "
                                      'reduces the risk of overfitting, '
                                      'leading to better performance on unseen '
                                      'data.'}]],
            'llm_output': None,
            'run': None}})
```



## LangSmith

所有`ChatModel`都内置了LangSmith跟踪功能。只需设置以下环境变量：

```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY=<你的API密钥>
```

然后，任何`ChatModel`调用（无论是否嵌套在链中）都将自动进行跟踪。跟踪将包括输入、输出、延迟、令牌使用情况、调用参数、环境参数等。在这里查看示例：https://smith.langchain.com/public/a54192ae-dd5c-4f7a-88d1-daa1eaba1af7/r。

在LangSmith中，您可以为任何跟踪提供反馈，为评估编译注释数据集，在游乐场中调试性能等。

## [Legacy] `__call__` {#legacy-**call**}

#### 消息输入 -> 消息输出

为了方便起见，您还可以将聊天模型视为可调用对象。可以通过将一个或多个消息传递给聊天模型来获取聊天完成。响应将是一个消息。

```python
from langchain.schema import HumanMessage, SystemMessage

chat(
    [
        HumanMessage(
            content="Translate this sentence from English to French: I love programming."
        )
    ]
)
```



```text
AIMessage(content="J'adore la programmation.")
```



OpenAI的聊天模型支持多条消息作为输入。有关更多信息，请参阅此处：https://platform.openai.com/docs/guides/chat/chat-vs-completions。以下是向聊天模型发送系统和用户消息的示例：

```python
messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(content="I love programming."),
]
chat(messages)
```



```text
AIMessage(content="J'adore la programmation.")
```



## [Legacy] `generate`

#### 批量调用，更丰富的输出

您可以更进一步，使用 `generate` 为多组消息生成完成项。这将返回一个带有附加 `message` 参数的 `LLMResult`。这将包括关于每个生成的额外信息，而不仅仅是返回的消息（例如完成原因），以及关于整个 API 调用的额外信息（例如使用的总令牌数）。

```python
batch_messages = [
    [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(content="I love programming."),
    ],
    [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(content="I love artificial intelligence."),
    ],
]
result = chat.generate(batch_messages)
result
```



```text
LLMResult(generations=[[ChatGeneration(text="J'adore programmer.", generation_info={'finish_reason': 'stop'}, message=AIMessage(content="J'adore programmer."))], [ChatGeneration(text="J'adore l'intelligence artificielle.", generation_info={'finish_reason': 'stop'}, message=AIMessage(content="J'adore l'intelligence artificielle."))]], llm_output={'token_usage': {'prompt_tokens': 53, 'completion_tokens': 18, 'total_tokens': 71}, 'model_name': 'gpt-3.5-turbo'}, run=[RunInfo(run_id=UUID('077917a9-026c-47c4-b308-77b37c3a3bfa')), RunInfo(run_id=UUID('0a70a0bf-c599-4f51-932a-c7d42202c984'))])
```



You can recover things like token usage from this LLMResult:

```python
result.llm_output
```



```text
{'token_usage': {'prompt_tokens': 53,
  'completion_tokens': 18,
  'total_tokens': 71},
 'model_name': 'gpt-3.5-turbo'}
```