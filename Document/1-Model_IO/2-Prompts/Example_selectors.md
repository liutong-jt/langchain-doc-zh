# Example selectors

如果您有大量的例子，可能需要选择哪些例子包含在提示中。例子选择器是负责这样做的类。

基本接口定义如下：

```python
class BaseExampleSelector(ABC):
    """Interface for selecting examples to include in prompts."""

    @abstractmethod
    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        
    @abstractmethod
    def add_example(self, example: Dict[str, str]) -> Any:
        """Add new example to store."""
```


它需要定义的唯一方法是 `select_examples` 方法。这个方法接受输入变量，然后返回一个例子列表。如何选择这些例子取决于每个特定实现。

LangChain有几种不同类型的例子选择器。要查看所有这些类型的概述，请参阅[此文档](https://python.langchain.com/docs/modules/model_io/prompts/example_selector_types)。

在这个指南中，我们将逐步创建一个自定义例子选择器。

## Examples

在这个指南中，我们将逐步创建一个自定义例子选择器。
为了使用示例选择器，我们需要创建一个示例列表。这些通常应该是示例输入和输出。出于本演示的目的，让我们想象我们正在选择如何将英语翻译成意大利语的示例。

```python
examples = [
    {"input": "hi", "output": "ciao"},
    {"input": "bye", "output": "arrivaderci"},
    {"input": "soccer", "output": "calcio"},
]
```

## Custom Example Selector

让我们写一个选择器，根据单词的长度选择要选择的例子。

```python
from langchain_core.example_selectors.base import BaseExampleSelector

class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        # This assumes knowledge that part of the input will be a 'text' key
        new_word = input_variables["input"]
        new_word_length = len(new_word)

        # Initialize variables to store the best match and its length difference
        best_match = None
        smallest_diff = float("inf")

        # Iterate through each example
        for example in self.examples:
            # Calculate the length difference with the first word of the example
            current_diff = abs(len(example["input"]) - new_word_length)

            # Update the best match if the current one is closer in length
            if current_diff < smallest_diff:
                smallest_diff = current_diff
                best_match = example

        return [best_match]
```

```python
example_selector = CustomExampleSelector(examples)
```

```python
example_selector.select_examples({"input": "okay"})
```

```text
[{'input': 'bye', 'output': 'arrivaderci'}]
```

```python
example_selector.add_example({"input": "hand", "output": "mano"})
```

```python
example_selector.select_examples({"input": "okay"})
```

```text
[{'input': 'hand', 'output': 'mano'}]
```

## Use in a Prompt

We can now use this example selector in a prompt

```python
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

example_prompt = PromptTemplate.from_template("Input: {input} -> Output: {output}")
```



```python
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Input: {input} -> Output:",
    prefix="Translate the following words from English to Italain:",
    input_variables=["input"],
)

print(prompt.format(input="word"))
```



```text
Translate the following words from English to Italain:

Input: hand -> Output: mano

Input: word -> Output:
```