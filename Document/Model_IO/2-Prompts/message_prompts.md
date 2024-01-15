# Types of `MessagePromptTemplate`

LangChain 提供了不同类型的 `MessagePromptTemplate`。最常用的是 `AIMessagePromptTemplate`、`SystemMessagePromptTemplate` 和 `HumanMessagePromptTemplate`，它们分别创建 AI 消息、系统消息和人类消息。

但是，在聊天模型支持任意角色的聊天消息的情况下，您可以使用 `ChatMessagePromptTemplate`，它允许用户指定角色名称。

```python
from langchain.prompts import ChatMessagePromptTemplate

prompt = "May the {subject} be with you"

chat_message_prompt = ChatMessagePromptTemplate.from_template(
    role="Jedi", template=prompt
)
chat_message_prompt.format(subject="force")
```

```text
ChatMessage(content='May the force be with you', role='Jedi')
```

但是，在聊天模型支持任意角色的聊天消息的情况下，您可以使用 `ChatMessagePromptTemplate`，它允许用户指定角色名称。
LangChain还提供了`MessagesPlaceholder`，它使您能够完全控制在格式化过程中要渲染的消息。当您不确定应该为消息提示模板使用什么角色，或者希望在格式
化过程中插入消息列表时，这可能会很有用。

```python
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

human_prompt = "Summarize our conversation so far in {word_count} words."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="conversation"), human_message_template]
)
```



```python
from langchain_core.messages import AIMessage, HumanMessage

human_message = HumanMessage(content="What is the best way to learn programming?")
ai_message = AIMessage(
    content="""\
1. Choose a programming language: Decide on a programming language that you want to learn.

2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.

3. Practice, practice, practice: The best way to learn programming is through hands-on experience\
"""
)

chat_prompt.format_prompt(
    conversation=[human_message, ai_message], word_count="10"
).to_messages()
```



```text
[HumanMessage(content='What is the best way to learn programming?'),
 AIMessage(content='1. Choose a programming language: Decide on a programming language that you want to learn.\n\n2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.\n\n3. Practice, practice, practice: The best way to learn programming is through hands-on experience'),
 HumanMessage(content='Summarize our conversation so far in 10 words.')]
```