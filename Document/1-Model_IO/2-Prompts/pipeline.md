# Pipeline

本笔记本将介绍如何组合多个提示。当您想要重用提示的某些部分时，这可能会很有用。这可以通过PipelinePrompt完成。PipelinePrompt由两个主要部分组成：

- 最终提示：返回的最终提示
- 管道提示：由字符串名称和提示模板组成的元组列表。每个提示模板将被格式化，然后作为具有相同名称的变量传递给未来的提示模板。

```python
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate
```



```python
full_template = """{introduction}

{example}

{start}"""
full_prompt = PromptTemplate.from_template(full_template)
```



```python
introduction_template = """You are impersonating {person}."""
introduction_prompt = PromptTemplate.from_template(introduction_template)
```



```python
example_template = """Here's an example of an interaction:

Q: {example_q}
A: {example_a}"""
example_prompt = PromptTemplate.from_template(example_template)
```



```python
start_template = """Now, do this for real!

Q: {input}
A:"""
start_prompt = PromptTemplate.from_template(start_template)
```



```python
input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt, pipeline_prompts=input_prompts
)
```



```python
pipeline_prompt.input_variables
```



```text
['example_q', 'example_a', 'input', 'person']
```



```python
print(
    pipeline_prompt.format(
        person="Elon Musk",
        example_q="What's your favorite car?",
        example_a="Tesla",
        input="What's your favorite social media site?",
    )
)
```



```text
You are impersonating Elon Musk.

Here's an example of an interaction:

Q: What's your favorite car?
A: Tesla

Now, do this for real!

Q: What's your favorite social media site?
A:
```