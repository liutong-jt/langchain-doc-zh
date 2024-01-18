# Quickstart

为了更好地理解代理框架，让我们构建一个具有两个工具的代理：一个用于在线查找，另一个用于查找我们加载到索引中的特定数据。
这将假设您已经了解了[LLMs](../model_io)和[检索](../data_connection)，如果您还没有探索这些部分，建议您这样做。

## Setup：LangSmith
根据定义，代理在返回面向用户的输出之前会采取自我决定的、依赖于输入的步骤序列。这使得调试这些系统特别棘手，可观察性尤为重要。[LangSmith](/docs/langsmith)在这种情况下特别有用。
使用LangChain构建时，所有步骤都将自动在LangSmith中进行跟踪。
要设置LangSmith，我们只需要设置以下环境变量：

```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="<your-api-key>"
```
## 定义工具
我们首先需要创建要使用的工具。我们将使用两个工具：[Tavily](/docs/integrations/tools/tavily_search)（用于在线搜索）和我们将在

### [Tavily](https://python.langchain.com/docs/integrations/tools/tavily_search)

我们在LangChain中内置了一个工具，可以轻松将Tavily搜索引擎用作工具。 请注意，这需要一个API密钥-他们有一个免费额度，但如果您没有或不想创建一个，您可以忽略此步骤。 创建API密钥后，您需要导出该密钥：

```bash
export TAVILY_API_KEY="..."
```



```python
from langchain_community.tools.tavily_search import TavilySearchResults
```



```python
search = TavilySearchResults()
```



```python
search.invoke("what is the weather in SF")
```



```text
[{'url': 'https://www.metoffice.gov.uk/weather/forecast/9q8yym8kr',
  'content': 'Thu 11 Jan Thu 11 Jan Seven day forecast for San Francisco  San Francisco (United States of America) weather Find a forecast  Sat 6 Jan Sat 6 Jan Sun 7 Jan Sun 7 Jan Mon 8 Jan Mon 8 Jan Tue 9 Jan Tue 9 Jan Wed 10 Jan Wed 10 Jan Thu 11 Jan  Find a forecast Please choose your location from the nearest places to : Forecast days Today Today Sat 6 Jan Sat 6 JanSan Francisco 7 day weather forecast including weather warnings, temperature, rain, wind, visibility, humidity and UV ... (11 January 2024) Time 00:00 01:00 02:00 03:00 04:00 05:00 06:00 07:00 08:00 09:00 10:00 11:00 12:00 ... Oakland Int. 11.5 miles; San Francisco International 11.5 miles; Corte Madera 12.3 miles; Redwood City 23.4 miles;'},
 {'url': 'https://www.latimes.com/travel/story/2024-01-11/east-brother-light-station-lighthouse-california',
  'content': "May 18, 2023  Jan. 4, 2024 Subscribe for unlimited accessSite Map Follow Us MORE FROM THE L.A. TIMES  Jan. 8, 2024 Travel & Experiences This must be Elysian Valley (a.k.a. Frogtown) Jan. 5, 2024 Food  June 30, 2023The East Brother Light Station in the San Francisco Bay is not a destination for everyone. ... Jan. 11, 2024 3 AM PT ... Champagne and hors d'oeuvres are served in late afternoon — outdoors if ..."}]
```



### Retriever
我们还将创建一个检索器，用于检索我们自己的某些数据。如需深入了解这里的每个步骤，请参阅 [Retrieval这一部分](/docs/modules/data_connection/)https://python.langchain.com/docs/modules/data_connection/)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()
```

```python
retriever.get_relevant_documents("how to upload a dataset")[0]
```

```text
Document(page_content="dataset uploading.Once we have a dataset, how can we use it to test changes to a prompt or chain? The most basic approach is to run the chain over the data points and visualize the outputs. Despite technological advancements, there still is no substitute for looking at outputs by eye. Currently, running the chain over the data points needs to be done client-side. The LangSmith client makes it easy to pull down a dataset and then run a chain over them, logging the results to a new project associated with the dataset. From there, you can review them. We've made it easy to assign feedback to runs and mark them as correct or incorrect directly in the web app, displaying aggregate statistics for each test project.We also make it easier to evaluate these runs. To that end, we've added a set of evaluators to the open-source LangChain library. These evaluators can be specified when initiating a test run and will evaluate the results once the test run completes. If we’re being honest, most of", metadata={'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith', 'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en'})
```

现在我们已经在我们要进行检索的索引中填充了内容，我们可以轻松地将其转换为一个工具（一个代理正确使用它的所需格式）。

```python
from langchain.tools.retriever import create_retriever_tool
```



```python
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
```



### 工具

现在我们已经创建了这两个，我们可以创建一个我们将在下游使用的工具列表。

```python
tools = [search, retriever_tool]
```



## Create the agent 创建代理

现在我们已经定义了工具，我们可以创建代理。我们将使用OpenAI Functions代理-有关此类代理的更多信息，以及其他选项，请参阅[本指南](https://python.langchain.com/docs/modules/agents/agent_types)
首先，我们选择要指导代理的LLM。

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```



Next, we choose the prompt we want to use to guide the agent.

If you want to see the contents of this prompt and have access to LangSmith, you can go to:

https://smith.langchain.com/hub/hwchase17/openai-functions-agent

```python
from langchain import hub

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages
```



```text
[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')),
 MessagesPlaceholder(variable_name='chat_history', optional=True),
 HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
 MessagesPlaceholder(variable_name='agent_scratchpad')]
```

现在，我们可以使用LLM、提示和工具来初始化代理。代理负责接收输入并决定采取什么行动。至关重要的是，代理不执行这些操作-这是由代理执行器（下一步）完成的。有关如何思考这些组件的更多信息，请参见我们的[概念指南agent-concepts](https://python.langchain.com/docs/modules/agents/concepts)

```python
from langchain.agents import create_openai_functions_agent

agent = create_openai_functions_agent(llm, tools, prompt)
```

最后,我们将智能体(大脑)与 AgentExecutor 中的工具结合在一起(AgentExecutor 会不断调用智能体并执行工具)。有关如何思考这些组件的更多信息，请参阅我们的[概念指南](./concepts)。https://python.langchain.com/docs/modules/agents/concepts)

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```



## Run the agent

我们现在可以在几个查询上运行代理！请注意，目前，这些查询都是**无状态**的（它不会记住以前的交互）。

```python
agent_executor.invoke({"input": "hi!"})
```



```text
> Entering new AgentExecutor chain...
Hello! How can I assist you today?

> Finished chain.
```



```text
{'input': 'hi!', 'output': 'Hello! How can I assist you today?'}
```



```python
agent_executor.invoke({"input": "how can langsmith help with testing?"})
```



```text
> Entering new AgentExecutor chain...

Invoking: `langsmith_search` with `{'query': 'LangSmith testing'}`


[Document(page_content='LangSmith 概述和用户指南 | 🦜️🛠️ LangSmith', metadata={'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith', 'description': '构建可靠的LLM应用程序可能具有挑战性。LangChain简化了初始设置，但仍需要工作来提高提示、链和代理的性能，使其达到足以在生产环境中使用水平。', 'language': 'en'}), Document(page_content='跳转到主要内容🦜️🛠️ LangSmith DocsPython DocsJS/TS DocsSearchGo到AppLangSmith概述追踪测试与评估组织中心LangSmith Cookbook概述在本页上LangSmith概述和用户指南构建可靠的LLM应用程序可能具有挑战性。LangChain简化了初始设置，但仍需要工作来提高提示、链和代理的性能，使其可靠到足以在生产中使用。在过去的两个月中，LangChain的我们一直在构建和使用LangSmith，目标是弥合这一差距。这是我们的战术用户指南，概述了有效使用LangSmith和最大化其好处的方法。默认开启\u200b在LangChain，我们所有人都默认在后台运行LangSmith的跟踪。在Python方面，这是通过设置环境变量实现的，我们每次启动虚拟环境或打开bash shell时都会建立这些环境变量，并保持设置状态。大多数JavaScript和TypeScript应用程序也是如此。', metadata={'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith', 'description': '构建可靠的LLM应用程序可能具有挑战性。LangChain简化了初始设置，但仍需要进行工作，以提高提示、链和代理的性能，使其达到足以在生产环境中使用的可靠程度。', 'language': 'en'}), Document(page_content='您还可以快速编辑示例并将其添加到数据集中，以扩展您的评估集的范围或微调模型以提高质量或降低成本。 监控 在完成所有这些操作后，您的应用程序可能最终准备投入生产。 LangSmith也可以用于以类似于您用于调试的方式监控您的应用程序。您可以记录所有痕迹，可视化延迟和令牌使用统计信息，并在问题出现时解决特定问题。每次运行还可以分配字符串标签或键值元数据，使您能够附加相关 id 或 AB 测试变体，并根据需要过滤运行。 我们还使您能够将反馈程序化地与运行关联。这意味着，如果您的应用程序上有赞/踩按钮，您可以使用它将反馈记录回 LangSmith。这可用于跟踪随时间推移的性能并找到表现不佳的数据点，您可以随后将这些数据点添加到数据集中以供将来测试——这与', metadata={'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith', 'description': '构建可靠的LLM应用程序可能具有挑战性。LangChain简化了初始设置，但仍需要进行工作，以提高提示、链和代理的性能，使其可靠到足以在生产中使用。', 'language': 'en'}), Document(page_content='翻译：输入，并观察会发生什么。但在某个时候，我们的应用程序运行良好，我们希望在测试更改时更加严谨。我们可以使用沿途构建的数据集（见上文）。或者，我们可以花一些时间手动构建一个小数据集。对于这些情况，LangSmith 简化', metadata={'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith', 'description': '构建可靠的LLM应用程序可能具有挑战性。LangChain简化了初始设置，但仍需要进行工作，以将提示、链和代理的性能提高到足以在生产环境中使用的水平。', 'language': 'en'})]
LangSmith 可以通过多种方式帮助进行测试。以下是 LangSmith 可以帮助测试的一些方式：

1. 跟踪：LangSmith 提供了跟踪功能，可用于在测试期间监控和调试应用程序。您可以记录所有跟踪，可视化延迟和令牌使用统计信息，并在出现问题时解决特定问题。

2. 评估：LangSmith 允许您快速编辑示例并将其添加到数据集中，以扩大评估集的表面积。这可以帮助您测试和微调模型，以提高质量或降低成本。

3. 监控：一旦应用程序准备就绪，可以使用 LangSmith 监控应用程序。您可以使用运行程序记录反馈，跟踪性能随时间的变化，并确定表现不佳的数据点。这些信息可用于改进应用程序并为未来的测试添加到数据集中。

4. 严格的测试：当您的应用程序运行良好并且想要对测试更改进行更严格的测试时，LangSmith 可以简化过程。您可以使用现有的数据集或手动构建小数据集来测试不同的场景并评估应用程序的性能。

要了解有关如何使用 LangSmith 进行测试的更详细信息，您可以参阅 [LangSmith 概述和用户指南](https://docs.smith.langchain.com/overview)。

> Finished chain.
```



```text
{'input': 'how can langsmith help with testing?',
 'output': 'LangSmith可以在多个方面帮助进行测试。以下是LangSmith在测试方面可以提供帮助的一些方法：

1. 跟踪：LangSmith提供了跟踪功能，可以在测试期间用于监控和调试应用程序。您可以记录所有跟踪，可视化延迟和令牌使用统计信息，并在出现问题时解决特定问题。

2. 评估：LangSmith允许您快速编辑示例并将它们添加到数据集中，以扩大评估集的表面积。这可以帮助您测试和微调模型，以提高质量或降低成本。

3. 监控：一旦您的应用程序准备就绪，可以投入生产，LangSmith可以用于监控您的应用程序。您可以使用运行程序记录反馈，跟踪性能随时间变化，并定位表现不佳的数据点。这些信息可用于改进您的应用程序，并为未来的测试添加到数据集中。

4. 严格的测试：当您的应用程序表现良好，并且想要更严格地测试更改时，LangSmith可以简化此过程。您可以使用现有数据集或手动构建小数据集来测试不同场景并评估应用程序的性能。

要了解更多有关如何使用LangSmith进行测试的详细信息，您可以参考[LangSmith概述和用户指南](https://docs.smith.langchain.com/overview)。'}
```



```python
agent_executor.invoke({"input": "whats the weather in sf?"})
```



```text
> Entering new AgentExecutor chain...

Invoking: `tavily_search_results_json` with `{'query': 'weather in San Francisco'}`


[{'url': 'https://www.whereandwhen.net/when/north-america/california/san-francisco-ca/january/', 'content': 'Best time to go to San Francisco? Weather in San Francisco in january 2024  How was the weather last january? Here is the day by day recorded weather in San Francisco in january 2023:  Seasonal average climate and temperature of San Francisco in january  8% 46% 29% 12% 8% Evolution of daily average temperature and precipitation in San Francisco in januaryWeather in San Francisco in january 2024. The weather in San Francisco in january comes from statistical datas on the past years. You can view the weather statistics the entire month, but also by using the tabs for the beginning, the middle and the end of the month. ... 11-01-2023 50°F to 54°F. 12-01-2023 50°F to 59°F. 13-01-2023 54°F to ...'}, {'url': 'https://www.latimes.com/travel/story/2024-01-11/east-brother-light-station-lighthouse-california', 'content': "May 18, 2023  Jan. 4, 2024 Subscribe for unlimited accessSite Map Follow Us MORE FROM THE L.A. TIMES  Jan. 8, 2024 Travel & Experiences This must be Elysian Valley (a.k.a. Frogtown) Jan. 5, 2024 Food  June 30, 2023The East Brother Light Station in the San Francisco Bay is not a destination for everyone. ... Jan. 11, 2024 3 AM PT ... Champagne and hors d'oeuvres are served in late afternoon — outdoors if ..."}]I'm sorry, I couldn't find the current weather in San Francisco. However, you can check the weather in San Francisco by visiting a reliable weather website or using a weather app on your phone.

> Finished chain.
```



```text
{'input': 'whats the weather in sf?',
 'output': "I'm sorry, I couldn't find the current weather in San Francisco. However, you can check the weather in San Francisco by visiting a reliable weather website or using a weather app on your phone."}
```



## Adding in memory

如前所述，该代理是无状态的。这意味着它不会记住之前的交互。为了给它添加内存支持，我们需要传递之前的`chat_history`。注意：由于我们使用的提示信息，它必须叫做`chat_history`。如果我们使用不同的提示信息，我们可以改变变量名称。

```python
# Here we pass in an empty list of messages for chat_history because it is the first message in the chat
agent_executor.invoke({"input": "hi! my name is bob", "chat_history": []})
```



```text
> Entering new AgentExecutor chain...
Hello Bob! How can I assist you today?

> Finished chain.
```



```text
{'input': 'hi! my name is bob',
 'chat_history': [],
 'output': 'Hello Bob! How can I assist you today?'}
```



```python
from langchain_core.messages import AIMessage, HumanMessage
```



```python
agent_executor.invoke(
    {
        "chat_history": [
            HumanMessage(content="hi! my name is bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
        ],
        "input": "what's my name?",
    }
)
```



```text
> Entering new AgentExecutor chain...
Your name is Bob.

> Finished chain.
```



```text
{'chat_history': [HumanMessage(content='hi! my name is bob'),
  AIMessage(content='Hello Bob! How can I assist you today?')],
 'input': "what's my name?",
 'output': 'Your name is Bob.'}
```



如果我们想要自动追踪这些消息，我们可以将其包装在 RunnableWithMessageHistory 中。关于如何使用它的更多信息，请参见 [this guide](https://python.langchain.com/docs/expression_language/how_to/message_history)。

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
```



```python
message_history = ChatMessageHistory()
```



```python
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
```



```python
agent_with_chat_history.invoke(
    {"input": "hi! I'm bob"},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}},
)
```



```text
> Entering new AgentExecutor chain...
Hello Bob! How can I assist you today?

> Finished chain.
```



```text
{'input': "hi! I'm bob",
 'chat_history': [],
 'output': 'Hello Bob! How can I assist you today?'}
```



```python
agent_with_chat_history.invoke(
    {"input": "what's my name?"},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}},
)
```



```text
> Entering new AgentExecutor chain...
Your name is Bob.

> Finished chain.
```



```text
{'input': "what's my name?",
 'chat_history': [HumanMessage(content="hi! I'm bob"),
  AIMessage(content='Hello Bob! How can I assist you today?')],
 'output': 'Your name is Bob.'}
```



## Conclusion

完工了！在这个快速入门中，我们介绍了如何创建一个简单的代理。代理是一个复杂的话题，有很多东西需要学习！返回到[主要代理页面](https://python.langchain.com/docs/modules/agents/)，可以找到更多关于概念指南、不同类型的代理、如何创建自定义工具等资源！

