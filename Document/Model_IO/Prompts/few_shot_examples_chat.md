## Few-shot examples for chat models

本笔记本介绍了如何在聊天模型中使用few-shot示例。似乎没有关于如何最好地进行few-shot提示的共识，而且最佳的提示编译方法可能会因模型而异。因此，我们提供了像 [FewShotChatMessagePromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.few_shot.FewShotChatMessagePromptTemplate.html?highlight=fewshot#langchain_core.prompts.few_shot.FewShotChatMessagePromptTemplate) 这样的少量提示模板作为灵活的起点，
您可以根据需要对其进行修改或替换。

少量提示模板的目标是根据输入动态选择示例，然后将示例格式化为最终提示以提供给模型。

**注意：** 以下代码示例适用于聊天模型。有关完成模型（LLMs）的类似少量提示示例，请参阅 [少量提示模板](https://python.langchain.com/docs/modules/model_io/prompts/few_shot_examples) 指南。

### Fixed Examples

最基本（也是最常见的）few-shot ptompting技术是使用固定提示示例。这样，您可以选择一个chain，评估它，避免在生产中担心额外的移动部件。

模板的基本组件包括：- `examples`：要包含在最终提示中的字典示例列表。- `example_prompt`：通过其 `format_messages` 方法将每个示例转换为 1 个或多个消息。常见的示例是将每个示例转换为一个human message和一个 AI message response，或者是一个human message 后跟一个function call message。

下面是一个简单的演示。首先，导入此示例所需的模块：

```python
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
```



Then, define the examples you’d like to include.

```python
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]
```



Next, assemble them into the few-shot prompt template.

```python
# This is a prompt template used to format each individual example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

print(few_shot_prompt.format())
```



```text
Human: 2+2
AI: 4
Human: 2+3
AI: 5
```



Finally, assemble your final prompt and use it with a model.

```python
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)
```



```python
from langchain_community.chat_models import ChatAnthropic

chain = final_prompt | ChatAnthropic(temperature=0.0)

chain.invoke({"input": "What's the square of a triangle?"})
```



```text
AIMessage(content=' Triangles do not have a "square". A square refers to a shape with 4 equal sides and 4 right angles. Triangles have 3 sides and 3 angles.\n\nThe area of a triangle can be calculated using the formula:\n\nA = 1/2 * b * h\n\nWhere:\n\nA is the area \nb is the base (the length of one of the sides)\nh is the height (the length from the base to the opposite vertex)\n\nSo the area depends on the specific dimensions of the triangle. There is no single "square of a triangle". The area can vary greatly depending on the base and height measurements.', additional_kwargs={}, example=False)
```



## Dynamic few-shot prompting

有时您可能希望根据输入条件显示哪些示例。为此，您可以将 `examples` 替换为 `example_selector`。其他组件与上面相同！回顾一下，动态少量示例提示
模板看起来像这样：

- `example_selector`：负责为给定输入选择少量示例（以及它们返回的顺序）。这些实现了 [BaseExampleSelector](https://api.python.langchain.com/en/latest/example_selectors/langchain_core.example_selectors.base.BaseExampleSelector.html?highlight=baseexampleselector#langchain_core.example_selectors.base.BaseExampleSelector) 接口。一个常见的例子是向量存储支持的 [SemanticSimilarityExampleSelector](https://api.python.langchain.com/en/latest/example_selectors/langchain_core.example_selectors.semantic_similarity.SemanticSimilarityExampleSelector.html?highlight=semanticsimilarityexampleselector#langchain_core.example_selectors.semantic_similarity.SemanticSimilarityExampleSelector)
- `example_prompt`：通过其 `format_messages` 方法将每个示例转换为 1 个或多个消息。一个常见的例子是将每个示例转换为一个人类消息和一个 AI 消息响应，或者一个人类消息后跟一个函数调用消息。

它们再次可以与其他消息和聊天模板组合，以组装您的最终提示。

```python
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
```

Since we are using a vectorstore to select examples based on semantic similarity, we will want to first populate the store.

```python
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "2+4", "output": "6"},
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?",
    },
]

to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)
```



#### Create the `example_selector`

With a vectorstore created, you can create the `example_selector`. Here we will isntruct it to only fetch the top 2 examples.

```python
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

# The prompt template will load examples by passing the input do the `select_examples` method
example_selector.select_examples({"input": "horse"})
```



```text
[{'input': 'What did the cow say to the moon?', 'output': 'nothing at all'},
 {'input': '2+4', 'output': '6'}]
```



#### Create prompt template

Assemble the prompt template, using the `example_selector` created above.

```python
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# Define the few-shot prompt.
few_shot_prompt = FewShotChatMessagePromptTemplate(
    # The input variables select the values to pass to the example_selector
    input_variables=["input"],
    example_selector=example_selector,
    # Define how each example will be formatted.
    # In this case, each example will become 2 messages:
    # 1 human, and 1 AI
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)
```



Below is an example of how this would be assembled.

```python
print(few_shot_prompt.format(input="What's 3+3?"))
```



```text
Human: 2+3
AI: 5
Human: 2+2
AI: 4
```



Assemble the final prompt template:

```python
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)
```



```python
print(few_shot_prompt.format(input="What's 3+3?"))
```



```text
Human: 2+3
AI: 5
Human: 2+2
AI: 4
```



#### Use with an LLM

Now, you can connect your model to the few-shot prompt.

```python
from langchain_community.chat_models import ChatAnthropic

chain = final_prompt | ChatAnthropic(temperature=0.0)

chain.invoke({"input": "What's 3+3?"})
```



```text
AIMessage(content=' 3 + 3 = 6', additional_kwargs={}, example=False)
```