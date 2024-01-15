# CSV parser

This output parser can be used when you want to return a list of comma-separated items.

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(temperature=0)

chain = prompt | model | output_parser
```



```python
chain.invoke({"subject": "ice cream flavors"})
```



```text
['Vanilla',
 'Chocolate',
 'Strawberry',
 'Mint Chocolate Chip',
 'Cookies and Cream']
```



```python
for s in chain.stream({"subject": "ice cream flavors"}):
    print(s)
```



```text
['Vanilla']
['Chocolate']
['Strawberry']
['Mint Chocolate Chip']
['Cookies and Cream']
```

# Datetime parser

This OutputParser can be used to parse LLM output into datetime format.

```python
from langchain.output_parsers import DatetimeOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
```



```python
output_parser = DatetimeOutputParser()
template = """Answer the users question:

{question}

{format_instructions}"""
prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)
```



```python
prompt
```



```text
PromptTemplate(input_variables=['question'], partial_variables={'format_instructions': "Write a datetime string that matches the following pattern: '%Y-%m-%dT%H:%M:%S.%fZ'.\n\nExamples: 0668-08-09T12:56:32.732651Z, 1213-06-23T21:01:36.868629Z, 0713-07-06T18:19:02.257488Z\n\nReturn ONLY this string, no other words!"}, template='Answer the users question:\n\n{question}\n\n{format_instructions}')
```



```python
chain = prompt | OpenAI() | output_parser
```



```python
output = chain.invoke({"question": "when was bitcoin founded?"})
```



```python
print(output)
```



```text
2009-01-03 18:15:05
```

# Enum parser

This notebook shows how to use an Enum output parser.

```python
from langchain.output_parsers.enum import EnumOutputParser
```



```python
from enum import Enum


class Colors(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
```



```python
parser = EnumOutputParser(enum=Colors)
```



```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template(
    """What color eyes does this person have?

> Person: {person}

Instructions: {instructions}"""
).partial(instructions=parser.get_format_instructions())
chain = prompt | ChatOpenAI() | parser
```



```python
chain.invoke({"person": "Frank Sinatra"})
```



```text
<Colors.BLUE: 'blue'>
```