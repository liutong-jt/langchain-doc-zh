# Example Selector Types

| Name       | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| Similarity | Uses semantic similarity between inputs and examples to decide which examples to choose. |
| MMR        | Uses Max Marginal Relevance between inputs and examples to decide which examples to choose. |
| Length     | Selects examples based on how many can fit within a certain length |
| Ngram      | Uses ngram overlap between inputs and examples to decide which examples to choose. |

# 示例选择器类型

| 名称   | 描述                                                   |
| ------ | ------------------------------------------------------ |
| 相似度 | 使用输入和示例之间的语义相似性来决定选择哪些示例。     |
| MMR    | 使用输入和示例之间的最大边际相关性来决定选择哪些示例。 |
| 长度   | 根据可以在特定长度内选择多少示例来选择示例。           |
| Ngram  | 使用输入和示例之间的ngram重叠来决定选择哪些示例。      |

## Select by length

这个示例选择器根据长度选择要使用的示例。当您担心构建的提示将超过上下文窗口的长度时，这很有用。对于较长的输入，它会选择较少的示例包含，而对于较短的输入，它会选择更多的示例。

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

# 创造反义词的假任务示例
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
example_selector = LengthBasedExampleSelector(
    # The examples it has available to choose from.
    examples=examples,
    # The PromptTemplate being used to format the examples.
    example_prompt=example_prompt,
    # 格式化后的示例应达到的最大长度。
    # 长度由下面的get_text_length函数测量。
    max_length=25,
    # 用于获取字符串长度的函数，
    # 用于确定应包含哪些示例。
    # 因为它提供了默认值，所以被注释掉了。
    # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
)
dynamic_prompt = FewShotPromptTemplate(
    # 我们提供了一个ExampleSelector，而不是示例。
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
```

```python
# 一个使用小输入的例子，因此它会选择所有示例。
print(dynamic_prompt.format(adjective="big"))
```

```text
Give the antonym of every input

Input: happy
Output: sad

Input: tall
Output: short

Input: energetic
Output: lethargic

Input: sunny
Output: gloomy

Input: windy
Output: calm

Input: big
Output:
```



```python
# 一个例子，输入很长，所以它只选择一个例子。
long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
print(dynamic_prompt.format(adjective=long_string))
```



```text
Give the antonym of every input

Input: happy
Output: sad

Input: big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else
Output:
```



```python
# You can add an example to an example selector as well.
new_example = {"input": "big", "output": "small"}
dynamic_prompt.example_selector.add_example(new_example)
print(dynamic_prompt.format(adjective="enthusiastic"))
```



```text
Give the antonym of every input

Input: happy
Output: sad

Input: tall
Output: short

Input: energetic
Output: lethargic

Input: sunny
Output: gloomy

Input: windy
Output: calm

Input: big
Output: small

Input: enthusiastic
Output:
```

## Select by maximal marginal relevance (MMR)

选择最大边际相关性（MMR）

`MaxMarginalRelevanceExampleSelector`根据输入最相似的例子以及优化多样性的组合来选择例子。它通过找到与输入具有最大余弦相似性的嵌入示例，然后
在对它们与已选择示例的接近程度进行惩罚的同时，迭代地将它们添加进来来实现这一点。

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]
```



```python
example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # 可供选择的示例列表。
    examples,
    # 用于生成用于衡量语义相似性的嵌入的嵌入类。
    OpenAIEmbeddings(),
    # 用于存储嵌入并向其进行相似性搜索的VectorStore类。
    FAISS,
    # 要生成的示例数量。
    k=2,
)
mmr_prompt = FewShotPromptTemplate(
    # 我们提供了一个ExampleSelector，而不是示例。
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
```



```python
# Input is a feeling, so should select the happy/sad example as the first one
print(mmr_prompt.format(adjective="worried"))
```



```text
Give the antonym of every input

Input: happy
Output: sad

Input: windy
Output: calm

Input: worried
Output:
```

```python
# 让我们将其与仅根据相似性得到的结果进行比较，
# 方法是使用SemanticSimilarityExampleSelector
# 而不是MaxMarginalRelevanceExampleSelector。
example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 可用于选择的示例列表。
    examples,
    # 用于生成用于测量语义相似性的嵌入的嵌入类。
    OpenAIEmbeddings(),
    # 用于存储嵌入并进行相似性搜索的VectorStore类。
    FAISS,
    # 要生成的示例数。
    k=2,
)
similar_prompt = FewShotPromptTemplate(
    # 我们提供了一个ExampleSelector，而不是示例。
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
print(similar_prompt.format(adjective="worried"))
```



```text
Give the antonym of every input

Input: happy
Output: sad

Input: sunny
Output: gloomy

Input: worried
Output:
```

## Select by n-gram overlap

根据ngram重叠度选择示例

`NGramOverlapExampleSelector`根据输入最相似的示例选择和排序示例。ngram重叠度是一个介于0.0和1.0之间的浮点数，包括0.0和1.0。

该选择器允许设置阈值分数。ngram重叠度分数小于或等于阈值的示例将被排除。默认情况下，阈值设置为-1.0，因此不会排除任何示例，只会重新排序它们。
将阈值设置为0.0将排除与输入没有任何ngram重叠的示例。

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Examples of a fictional translation task.
examples = [
    {"input": "See Spot run.", "output": "Ver correr a Spot."},
    {"input": "My dog barks.", "output": "Mi perro ladra."},
    {"input": "Spot can run.", "output": "Spot puede correr."},
]
```



```python
example_selector = NGramOverlapExampleSelector(
    # The examples it has available to choose from.
    examples=examples,
    # The PromptTemplate being used to format the examples.
    example_prompt=example_prompt,
    # The threshold, at which selector stops.
    # It is set to -1.0 by default.
    threshold=-1.0,
    # For negative threshold:
    # Selector sorts examples by ngram overlap score, and excludes none.
    # For threshold greater than 1.0:
    # Selector excludes all examples, and returns an empty list.
    # For threshold equal to 0.0:
    # Selector sorts examples by ngram overlap score,
    # and excludes those with no ngram overlap with input.
)
dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the Spanish translation of every input",
    suffix="Input: {sentence}\nOutput:",
    input_variables=["sentence"],
)
```



```python
# An example input with large ngram overlap with "Spot can run."
# and no overlap with "My dog barks."
print(dynamic_prompt.format(sentence="Spot can run fast."))
```



```text
Give the Spanish translation of every input

Input: Spot can run.
Output: Spot puede correr.

Input: See Spot run.
Output: Ver correr a Spot.

Input: My dog barks.
Output: Mi perro ladra.

Input: Spot can run fast.
Output:
```



```python
# You can add examples to NGramOverlapExampleSelector as well.
new_example = {"input": "Spot plays fetch.", "output": "Spot juega a buscar."}

example_selector.add_example(new_example)
print(dynamic_prompt.format(sentence="Spot can run fast."))
```



```text
Give the Spanish translation of every input

Input: Spot can run.
Output: Spot puede correr.

Input: See Spot run.
Output: Ver correr a Spot.

Input: Spot plays fetch.
Output: Spot juega a buscar.

Input: My dog barks.
Output: Mi perro ladra.

Input: Spot can run fast.
Output:
```



```python
# You can set a threshold at which examples are excluded.
# For example, setting threshold equal to 0.0
# excludes examples with no ngram overlaps with input.
# Since "My dog barks." has no ngram overlaps with "Spot can run fast."
# it is excluded.
example_selector.threshold = 0.0
print(dynamic_prompt.format(sentence="Spot can run fast."))
```



```text
Give the Spanish translation of every input

Input: Spot can run.
Output: Spot puede correr.

Input: See Spot run.
Output: Ver correr a Spot.

Input: Spot plays fetch.
Output: Spot juega a buscar.

Input: Spot can run fast.
Output:
```



```python
# Setting small nonzero threshold
example_selector.threshold = 0.09
print(dynamic_prompt.format(sentence="Spot can play fetch."))
```



```text
Give the Spanish translation of every input

Input: Spot can run.
Output: Spot puede correr.

Input: Spot plays fetch.
Output: Spot juega a buscar.

Input: Spot can play fetch.
Output:
```



```python
# Setting threshold greater than 1.0
example_selector.threshold = 1.0 + 1e-9
print(dynamic_prompt.format(sentence="Spot can play fetch."))
```



```text
Give the Spanish translation of every input

Input: Spot can play fetch.
Output:
```

## Select by similarity

这个对象根据与输入的相似度选择示例。它通过查找嵌入具有与输入最大的余弦相似度的示例来实现这一点。

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]
```



```python
example_selector = SemanticSimilarityExampleSelector.from_examples(
    # The list of examples available to select from.
    examples,
    # The embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # The VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # The number of examples to produce.
    k=1,
)
similar_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
```



```python
# Input is a feeling, so should select the happy/sad example
print(similar_prompt.format(adjective="worried"))
```



```text
Give the antonym of every input

Input: happy
Output: sad

Input: worried
Output:
```



```python
# Input is a measurement, so should select the tall/short example
print(similar_prompt.format(adjective="large"))
```



```text
Give the antonym of every input

Input: tall
Output: short

Input: large
Output:
```



```python
# You can add new examples to the SemanticSimilarityExampleSelector as well
similar_prompt.example_selector.add_example(
    {"input": "enthusiastic", "output": "apathetic"}
)
print(similar_prompt.format(adjective="passionate"))
```



```text
Give the antonym of every input

Input: enthusiastic
Output: apathetic

Input: passionate
Output:
```