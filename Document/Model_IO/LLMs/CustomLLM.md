# 自定义LLM

本笔记本介绍了如何创建自定义LLM包装器，以便在您想使用自己的LLM或使用LangChain中不支持的不同包装器时使用。

自定义LLM需要实现的只有两件事：

- 一个`_call`方法，该方法接受一个字符串，一些可选的停止词，并返回一个字符串。
- 一个`_llm_type`属性，该属性返回一个字符串。仅用于日志记录。

它可以实现的第二个可选事项是：

- 一个`_identifying_params`属性，用于帮助打印此类。应返回一个字典。

让我们实现一个非常简单的自定义LLM，该LLM只返回输入的前n个字符。

```python
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
```



```python
class CustomLLM(LLM):
    n: int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
```



We can now use this as an any other LLM.

```python
llm = CustomLLM(n=10)
```

```python
llm.invoke("This is a foobar thing")
```

```text
'This is a '
```

We can also print the LLM and see its custom print.

```python
print(llm)
```



```text
CustomLLM
Params: {'n': 10}
```