# Langchain API

## Prompt

| langchain        | Method           |                                                           |
| ---------------- | ---------------- | --------------------------------------------------------- |
| `PromptTemplate` |                  | input_variables=['key1', 'key2']<br />template: str=''    |
|                  | format()         | key1='', key2=''                                          |
|                  | from _template() | template 无需加入input_variables， 该方法会自动寻找关键字 |
|                  | input_variables  | 输出关键字                                                |

| langchain.prompt              | Method                    | comment                                                  |
| ----------------------------- | ------------------------- | -------------------------------------------------------- |
| `ChatPromptTemplate`          | from_message()            | 可以包含多个MessagePrompt                                |
|                               | format_prompt()           | 返回一个`PromptValue`, 他可以转换一个字符串和Message对象 |
|                               | PromptValue.to_messages() | 转化成messages对象                                       |
| `PromptTemplate`              |                           |                                                          |
| `SystemMessagePromptTemplate` | from_template()           | 和PromptTemplate方法相同                                 |
| `AIMessagePromptTemplate`     | from_template()           |                                                          |
| `HumanMessagePromptTemplate`  | from_template()           |                                                          |
|                               |                           |                                                          |

| langchain.schema | Method | comment |
| ---------------- | ------ | ------- |
| `AIMessage`      |        |         |
| `HumanMessage`   |        |         |
| `SystemMessage`  |        |         |

