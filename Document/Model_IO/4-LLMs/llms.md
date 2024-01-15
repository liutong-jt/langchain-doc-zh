# LLMs

大型语言模型（LLMs）是LangChain的核心组成部分。LangChain并不提供自己的LLMs，而是提供了一个标准接口，用于与许多不同的LLMs交互。具体来说，这个接
口是一个输入为字符串并返回字符串的接口。
有很多LLMs提供商（OpenAI、Cohere、Hugging Face等）——`LLM`类旨在为所有这些提供商提供一个标准接口。
## 快速入门
查看[此快速入门](https://python.langchain.com/docs/modules/model_io/llms/quick_start)，了解如何使用LLMs，包括它们提供的所有不同方法的概述。
## 集成
要查看LangChain提供的所有LLM集成的完整列表，请转到[Integrations页面](https://python.langchain.com/docs/integrations/llms/)。
## 如何使用指南
我们有几个关于更高级的LLM使用方法的如何使用指南。这包括：
- [如何编写自定义LLM类](https://python.langchain.com/docs/modules/model_io/llms/custom_llm)
- [如何缓存LLM响应](https://python.langchain.com/docs/modules/model_io/llms/llm_caching)
- [如何从LLM流式传输响应](https://python.langchain.com/docs/modules/model_io/llms/streaming_llm)
- [如何在LLM调用中跟踪令牌使用情况）（./token_usage_tracking）