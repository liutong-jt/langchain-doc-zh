# Model I/O
任何语言模型的核心都是模型。 `LangChain`可以对任何模型构建接口。  
- Prompts : 模板，动态化选择，管理模型输出
- Language models : 通过公共接口调用语言模型
- Output parsers: 从模型的输出中提取信息  

![model_io_diagram](https://python.langchain.com/assets/images/model_io-1f23a36233d7731e93576d6885da2750.jpg)

## Prompts
通过Prompts编程是一种新的编程方式。 **Promps**指的是(refers to) 对大模型的输入。这个输入通常被多个小组件共同构建。Langchain 提供了数几种`classes`和`funciton`来使构建和使用prompt更简单。

- Prompt templates: 模型输入的参数化
- Example selectors: 动态选择要包含在提示中的示例

### prompt 模板

语言模型需要一个文本作为输入 -- 这个输入一般被是为prompt。通常来说，这不是一个简单的硬编码的字符串，而是一个模板的组合，例如，用户的输入。Langchain提供了书记个`classes`和`functions`来使构建和使用prompts更简单。

#### 什么是 prompt template?

---

prompt template 是一个可重复的**(reproducible)**方式来生成一个prompt。它包含了一个文本字符串("the template"),  然后在用户输入结束后传入一系列参数生成一个prompt。

一个prompt template可以包括：

- 对语言模型的指令(instruction)，
- 一系列简短的例子来帮助语言模型生成一个更好的response，
- 对语言模型提出一个问题。

这有一个简答的例子

```python
from langchain import PromptTemplate

template = """\
你是一个新的命名咨询公司。
一个制作{product}的公司可以取一个什么样的好名字？
"""

prompt = PromptTemplate.from_template(template)
prompt.format(product="多彩袜子")
```

```text
输出：
'你是一个新的命名咨询公司。\n一个制作彩色袜子的公司可以取一个什么样的好名字?\n'
```

#### 创建一个prompt template

---

你可以通过使用`PromptTemplate`类创建一个简单hardcode的prompt。prompt templates能够接受任何数量的输入变量，还可以格式化的方式生成prompt。

```python
from langchain import PromptTemplate

# 一个没有输入变量的例子
no_input_prompt = PromptTemplate(input_variables=[], 
                                  template="Tell me a joke.")
no_input_prompt.format()
# -> "Tell me a joke"

# 有一个输入的例子
one_input_prompt = PromptTemplate(input_variables=['adjective'],
                                  template="Tell me a {adjective} joke.")
one_input_prompt.format(adjective="funny")
# -> "Tell me a funny joke."

# 有多个输入的例子
multiple_input_prompt = PromptTemplate(
    input_variables=["adjective", "content"],
    template="Tell me a {adjective} joke about {content}")
multiple_input_prompt.format(adjective='funny', content='chickens')
# -> "Tell me a funny joke about chickens."
```

如果你不想要手动指定`input_variables`， 你也可以创建`PromptTemplate`通过使用`from_template`类方法。`langchain`可以基于`template`方法自动推断出`input_variables`的输入变量。

```python
template = 'Tell me a {adjective} joke about {content}'

prompt_template = PromptTemplate.from_template(template)
prompt_template.input_variables
# -> ['adjective', 'content']
prompt_template.format(adjective='funny', content='chickens')
# -> 'Tell me a funny joke about chickens'
```

你可以创建自定义模板，并以任何你想要的方式对其格式化。关于更多的信息，参考[Custome Prompte Templates](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/custom_prompt_template.html).

#### Chat prompt template

---

[Chat Models]()接受一个聊天消息列表作为输入 - 这个列表一般指的是一个`prompt`。 这些聊天消息与原始字符串 (你想要传递给LLM模型) 不同的是，每条消息都有一个相关的`role`。

例如，在OpenAI [Chat Completion API](https://platform.openai.com/docs/guides/chat/introduction)中， 一个聊天消息可以与AI，人类或者系统角色相关。这个模型应该更紧密的遵循系统消聊天信息的提示。

LangChain提供了数几个prompt template, 以便更简单地去构建和使用prompt。我们鼓励你使用这些聊天相关的prompt来代替`PromptTemplate` ，在当询问聊天模型时，可以更加充分挖掘出底层聊天模型的潜力。（when querying chat models to **fully exploit** the **potential of underlying** chat model.）

```python
from langchain.prompts import (
	ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
	AIMessage, 
    HumanMessage,
    SystemMessage
)
```

为了创建一个与角色相关的消息模板，你应该使用`MessagePromptTemplate`.

为了更方便，template里有一个`from_template`方法。如果你使用这个template， 它就会像这样：

```python
template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_mesage_prompt = HumanMessagePromptTempalte.from_template(human_template)
```

如果你想要更直接构建`MessagePromptTemplate`，你可以在外边创建一个`PromptTemplate`然后传递他进来，例如：

```python
prompt = PromptTemplate(
	template='You are a helpful assistant that translate {input_language} to {output_language}.',
    input_variables=['input_language', 'output_language'],
)
system_message_prompt_2 = SystemMessagePromptTemplage(prompt=prompt)

assert system_message_prompt == system_message_prompt_2
```

在这之后，你可以从一个到更多的`MessagePromptTemplate`构建一个`ChatPromptTemplate`。 你可以使用`ChatPromptTemplate`的`format_prompt`方法 -- 他返回一个`PromptValue`, 这个你可以转化到一个字符串和一个Message对象， 这取决你想要使用格式化的值作为一个输入传递给LLM或者聊天模型。

```python
chat_prompt = ChatPromptTemplate.from_message([system_message_prompt, human_message_prompt])

# get a chat completion from the ormatted messages
chat_prompt.format_prompt(input_language='English', 
                          output_language='Franch', 
                          text='I love programming.'
                     ).to_messages()
```

```text
    [SystemMessage(content='You are a helpful assistant that translates English to French.', additional_kwargs={}),
     HumanMessage(content='I love programming.', additional_kwargs={})]
```

### Example selectors

如果你有很多数量的例子，你可能需要选择哪一个包括在prompt中。这个`Example Selector`类可以帮你做这件事。

基础的接口定义如下：

```python
class BaseExampleSelector(ABC):
    """Interface for selecting examples to include in prompts.
    	选择一个例子包含在prompt的方法接口"""
    
    @abstractmethod
    def select_example(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the input.
        	基于输入来选择用哪个例子"""
```

他唯一需要被公开(expose)的方法是`select_examples`方法。它接受输入变量然后返回一个examples的list。~~并且它的特异性取决于接口如何选择example~~至于如何选择这些例子，则取决于每个具体的接口(It is up to each specific implementation as to how those examples are selected.)。让我们看看下边的例子。

#### Custom example selector

在本教程中，我们将创建一个自定义的example selector选择器，从给出的了examples列表中选择出每个备用的例子(that selects every alternate example from a given list of examples.)。

`ExampleSelector`必须完成以下两个接口方法：

1. `add_example`接受一个example并且将其添加到Example Selector
2. `select_example`接受一个输入变量(用户的输入)，然后返回一个在数几个few shot prompt中使用的examples列表。

#### Select by length

#### Select by maximal marginal relevance (MMR)

#### Select by n-gram overlap

#### Select by similarity

## Language models

## Output parsers

# Data connection