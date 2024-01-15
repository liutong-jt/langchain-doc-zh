# 聊天模型
聊天模型是LangChain的核心组件之一。LangChain不提供自己的聊天模型，而是提供了与多种不同模型交互的标准接口。具体来说，该接口是一个接受消息列表作
为输入并返回消息的接口。
有许多模型提供商（OpenAI、Cohere、Hugging Face等），`ChatModel`类旨在为所有这些提供标准接口。
## 快速入门
查看[此快速入门](https://python.langchain.com/docs/modules/model_io/chat/quick_start)，以了解如何使用聊天模型，包括它们提供的所有不同方法。
## 集成
要查看LangChain提供的所有LLM集成的完整列表，请转到[Integrations页面](https://python.langchain.com/docs/integrations/chat/)。
## 如何使用指南
我们有几个关于更高级的LLM使用方法的如何使用指南。这包括：

- [如何缓存ChatModel响应](https://python.langchain.com/docs/modules/model_io/chat/chat_model_caching)
- [如何从ChatModel流式传输响应](https://python.langchain.com/docs/modules/model_io/chat/streaming)
- [如何在ChatModel调用中跟踪令牌使用情况](https://python.langchain.com/docs/modules/model_io/chat/token_usage)