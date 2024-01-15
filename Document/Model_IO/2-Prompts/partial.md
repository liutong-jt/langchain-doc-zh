# Partial prompt templates

与其它方法一样，可以对提示模板进行“部分”处理，例如，传入所需值的一个子集，以创建一个新的提示模板，该模板仅期望剩余的子集值。

LangChain 通过两种方式支持此功能：1. 使用字符串值的部分格式化。 2. 使用返回字符串值的函数的部分格式化。

这两种不同的方式支持不同的用例。在下面的示例中，我们将介绍这两种用例的动机以及如何在 LangChain 中实现它们。

## Partial with strings使用字符串部分

想要对提示模板进行部分处理的一个常见用例是，您在获取其它变量之前获取某些变量。例如，假设您有一个需要两个变量 `foo` 和 `baz` 的提示模板。如果在chain的早期阶段就获取了 `foo` 值，但稍后才获取 `baz` 值，则在等待将两个变量放在同一位置以便传递给提示模板时可能会很烦人。相反，您可以使用 `foo` 值部分处理提示模板，然后将部分处理后的提示模板传递下去并仅使用该模板。以下是一个示例：

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(template="{foo}{bar}", input_variables=["foo", "bar"])
partial_prompt = prompt.partial(foo="foo")
print(partial_prompt.format(bar="baz"))
```



```text
foobaz
```



你也可以直接使用部分变量来初始化提示。

```python
prompt = PromptTemplate(
    template="{foo}{bar}", input_variables=["bar"], partial_variables={"foo": "foo"}
)
print(prompt.format(bar="baz"))
```



```text
foobaz
```



## Partial with functions带函数的偏函数

另一种常见的用途是使用函数偏函数。这种情况的使用场景是，当你有一个变量，你总是想要以一种常见的方式获取它。这方面的最佳例子是日期或时间。想象一下，你总是想要在提示中包含当前日期。你不能在提示中硬编码它，而与其他输入变量一起传递它有点烦人。在这种情况下，能够使用始终返回当前日期的函数作为偏函数提示是非常方便的。

```python
from datetime import datetime

def _get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")
```



```python
prompt = PromptTemplate(
    template="Tell me a {adjective} joke about the day {date}",
    input_variables=["adjective", "date"],
)
partial_prompt = prompt.partial(date=_get_datetime)
print(partial_prompt.format(adjective="funny"))
```



```text
Tell me a funny joke about the day 12/27/2023, 10:45:22
```

您还可以仅使用部分变量初始化提示，这在这种工作流程中通常更有意义。

```python
prompt = PromptTemplate(
    template="Tell me a {adjective} joke about the day {date}",
    input_variables=["adjective"],
    partial_variables={"date": 
                      },
)
print(prompt.format(adjective="funny"))
```



```text
Tell me a funny joke about the day 12/27/2023, 10:45:36
```