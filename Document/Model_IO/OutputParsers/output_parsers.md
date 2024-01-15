# 输出解析器

输出解析器负责将LLM的输出转换为更合适的格式。当你让LLM生成任何形式的结构化数据时，这非常有用。

除了拥有大量不同类型的输出解析器外，LangChain OutputParsers的一个独特优势是其中许多支持流式处理。

## 快速入门

请参阅[此快速入门指南](https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start)，了解输出解析器的介绍以及如何使用它们。

## 输出解析器类型

LangChain有许多不同类型的输出解析器。这是一个LangChain支持的输出解析器列表。下表包含各种信息：

**名称**：输出解析器的名称

**支持流式处理**：输出解析器是否支持流式处理。

**具有格式说明**：输出解析器是否具有格式说明。这通常可用，除非（a）提示中未指定所需的模式，而是通过其他参数（如OpenAI函数调用）指定，或者（b）OutputParser包装了另一个OutputParser。

**调用LLM**：此输出解析器本身是否调用LLM。这通常仅由尝试纠正格式错误的输出解析器完成。

**输入类型**：预期的输入类型。大多数输出解析器都可以处理字符串和消息，但有些（如OpenAI Functions）需要具有特定kwargs的消息。

**输出类型**：解析器返回的对象的输出类型。

**描述**：我们对此输出解析器的评论以及何时使用它。

| Name                                                         | Supports Streaming | Has Format Instructions       | Calls LLM | Input Type                       | Output Type          | Description                                                  |      |      |
| ------------------------------------------------------------ | ------------------ | ----------------------------- | --------- | -------------------------------- | -------------------- | ------------------------------------------------------------ | ---- | ---- |
| [OpenAIFunctions](https://python.langchain.com/docs/modules/model_io/output_parsers/types/openai_functions) | ✅                  | (Passes `functions` to model) |           | `Message` (with `function_call`) | JSON object          | Uses OpenAI function calling to structure the return output. If you are using a model that supports function calling, this is generally the most reliable method. |      |      |
| [JSON](https://python.langchain.com/docs/modules/model_io/output_parsers/types/json) | ✅                  | ✅                             |           | `str \| Message`                 | JSON object          | Returns a JSON object as specified. You can specify a Pydantic model and it will return JSON for that model. Probably the most reliable output parser for getting structured data that does NOT use function calling. |      |      |
| [XML](https://python.langchain.com/docs/modules/model_io/output_parsers/types/xml) | ✅                  | ✅                             |           | `str \| Message`                 | `dict`               | Returns a dictionary of tags. Use when XML output is needed. Use with models that are good at writing XML (like Anthropic's). |      |      |
| [CSV](https://python.langchain.com/docs/modules/model_io/output_parsers/types/csv) | ✅                  | ✅                             |           | `str \| Message`                 | `List[str]`          | Returns a list of comma separated values.                    |      |      |
| [OutputFixing](https://python.langchain.com/docs/modules/model_io/output_parsers/types/output_fixing) |                    |                               | ✅         | `str \| Message`                 |                      | Wraps another output parser. If that output parser errors, then this will pass the error message and the bad output to an LLM and ask it to fix the output. |      |      |
| [RetryWithError](https://python.langchain.com/docs/modules/model_io/output_parsers/types/retry) |                    |                               | ✅         | `str \| Message`                 |                      | Wraps another output parser. If that output parser errors, then this will pass the original inputs, the bad output, and the error message to an LLM and ask it to fix it. Compared to OutputFixingParser, this one also sends the original instructions. |      |      |
| [Pydantic](https://python.langchain.com/docs/modules/model_io/output_parsers/types/pydantic) |                    | ✅                             |           | `str \| Message`                 | `pydantic.BaseModel` | Takes a user defined Pydantic model and returns data in that format. |      |      |
| [YAML](https://python.langchain.com/docs/modules/model_io/output_parsers/types/yaml) |                    | ✅                             |           | `str \| Message`                 | `pydantic.BaseModel` | Takes a user defined Pydantic model and returns data in that format. Uses YAML to encode it. |      |      |
| [PandasDataFrame](https://python.langchain.com/docs/modules/model_io/output_parsers/types/pandas_dataframe) |                    | ✅                             |           | `str \| Message`                 | `dict`               | Useful for doing operations with pandas DataFrames.          |      |      |
| [Enum](https://python.langchain.com/docs/modules/model_io/output_parsers/types/enum) |                    | ✅                             |           | `str \| Message`                 | `Enum`               | Parses response into one of the provided enum values.        |      |      |
| [Datetime](https://python.langchain.com/docs/modules/model_io/output_parsers/types/datetime) |                    | ✅                             |           | `str \| Message`                 | `datetime.datetime`  | Parses response into a datetime string.                      |      |      |
| [Structured](https://python.langchain.com/docs/modules/model_io/output_parsers/types/structured) |                    | ✅                             |           | `str \| Message`                 | `Dict[str, str]`     | An output parser that returns structured information. It is less powerful than other output parsers since it only allows for fields to be strings. This can be useful when you are working with smaller LLMs. |      |      |