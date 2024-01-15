# 通过标记分割

语言模型具有标记限制。您不应超过标记限制。因此，当您将文本分割成块时，计算标记数是一个好主意。有许多分词器。在计算文本中的标记数时，应使用与语言模型中使用的相同的分词器。

## tiktoken

> [tiktoken](https://github.com/openai/tiktoken) 是由 `OpenAI` 创建的快速 `BPE` 分词器。

我们可以使用它来估计使用的标记数。对于 OpenAI 模型，它可能更准确。

1. 文本是如何分割的：通过传入的字符。
2. 块大小是如何测量的：通过 `tiktoken` 分词器。

```python
#!pip install tiktoken
```



```python
# This is a long document we can split up.
with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()
from langchain.text_splitter import CharacterTextSplitter
```



```python
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(state_of_the_union)
```



```python
print(texts[0])
```



```text
Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  

Last year COVID-19 kept us apart. This year we are finally together again. 

Tonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. 

With a duty to one another to the American people to the Constitution.
```



注意，如果我们使用 `CharacterTextSplitter.from_tiktoken_encoder`，文本只会被 `CharacterTextSplitter` 和 `tiktoken` 分词器分割。这意味着分割可能会大于由 `tiktoken` 分词器测
量的块大小。我们可以使用 `RecursiveCharacterTextSplitter.from_tiktoken_encoder` 来确保分割不会大于语言模型允许的令牌块大小，其中每个分割如果大小较大则会被递归地分割。

我们也可以直接加载一个 tiktoken 分割器，以确保每个分割都小于块大小。

```python
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

texts = text_splitter.split_text(state_of_the_union)
print(texts[0])
```



## spaCy

[spaCy](https://spacy.io/) 是一个开源的软件库，用于高级自然语言处理，使用 Python 和 Cython 编程语言编写。

另一个与 `NLTK` 相似的替代方案是使用 [spaCy 分词器](https://spacy.io/api/tokenizer)。

1. 文本如何分割：由 spaCy 分词器分割。
2. 分块大小如何测量：通过字符数量测量。

```python
#!pip install spacy
```



```python
# This is a long document we can split up.
with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()
```



```python
from langchain.text_splitter import SpacyTextSplitter

text_splitter = SpacyTextSplitter(chunk_size=1000)
```



```python
texts = text_splitter.split_text(state_of_the_union)
print(texts[0])
```



```text
Madam Speaker, Madam Vice President, our First Lady and Second Gentleman.

Members of Congress and the Cabinet.

Justices of the Supreme Court.

My fellow Americans.  



Last year COVID-19 kept us apart.

This year we are finally together again. 



Tonight, we meet as Democrats Republicans and Independents.

But most importantly as Americans. 



With a duty to one another to the American people to the Constitution. 



And with an unwavering resolve that freedom will always triumph over tyranny. 



Six days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways.

But he badly miscalculated. 



He thought he could roll into Ukraine and the world would roll over.

Instead he met a wall of strength he never imagined. 



He met the Ukrainian people. 



From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world.
```



## SentenceTransformers 句子转换器 

> `SentenceTransformersTokenTextSplitter` 是一个用于与句子转换器模型一起使用的特殊文本分隔器。默认行为是将文本分割成适合您想要使用的句子转换器模型的令牌窗口的块。



```python
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
```



```python
splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)
text = "Lorem "
```



```python
count_start_and_stop_tokens = 2
text_token_count = splitter.count_tokens(text=text) - count_start_and_stop_tokens
print(text_token_count)
```



```text
2
```



```python
token_multiplier = splitter.maximum_tokens_per_chunk // text_token_count + 1

# `text_to_split` does not fit in a single chunk
text_to_split = text * token_multiplier

print(f"tokens in text to split: {splitter.count_tokens(text=text_to_split)}")
```



```text
tokens in text to split: 514
```



```python
text_chunks = splitter.split_text(text=text_to_split)

print(text_chunks[1])
```



```text
lorem
```



## NLTK

> [自然语言工具包](https://en.wikipedia.org/wiki/Natural_Language_Toolkit)，或通常被称为 [NLTK](https://www.nltk.org/)，是一个用于英语符号和统计自然语言处理（NLP）的Python编程语言库和程序套件。我们可以通过NLTK分词器进行分词，而不是仅按" "进行分割。1. 分词方式：通过NLTK分词器。2. 分块大小测量方式：通过字符数。

```python
# pip install nltk
```



```python
# This is a long document we can split up.
with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()
```



```python
from langchain.text_splitter import NLTKTextSplitter

text_splitter = NLTKTextSplitter(chunk_size=1000)
```



```python
texts = text_splitter.split_text(state_of_the_union)
print(texts[0])
```



```text
Madam Speaker, Madam Vice President, our First Lady and Second Gentleman.

Members of Congress and the Cabinet.

Justices of the Supreme Court.

My fellow Americans.

Last year COVID-19 kept us apart.

This year we are finally together again.

Tonight, we meet as Democrats Republicans and Independents.

But most importantly as Americans.

With a duty to one another to the American people to the Constitution.

And with an unwavering resolve that freedom will always triumph over tyranny.

Six days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways.

But he badly miscalculated.

He thought he could roll into Ukraine and the world would roll over.

Instead he met a wall of strength he never imagined.

He met the Ukrainian people.

From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world.

Groups of citizens blocking tanks with their bodies.
```



## Hugging Face 分词器

> [Hugging Face](https://huggingface.co/docs/tokenizers/index) 提供了许多分词器。

我们使用 Hugging Face 分词器，[GPT2TokenizerFast](https://huggingface.co/Ransaka/gpt2-tokenizer-fast) 来计算文本的 token 长度。

1. 文本如何分割：通过传入的字符进行分割。
2. 块大小如何测量：通过 `Hugging Face` 分词器计算的 token 数量。

```python
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
```



```python
# This is a long document we can split up.
with open("../../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()
from langchain.text_splitter import CharacterTextSplitter
```



```python
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(state_of_the_union)
```



```python
print(texts[0])
```



```text
Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  

Last year COVID-19 kept us apart. This year we are finally together again. 

Tonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. 

With a duty to one another to the American people to the Constitution.
```