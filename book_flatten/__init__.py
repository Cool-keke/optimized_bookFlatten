# book_flatten/__init__.py

# 导入、导出 correct_paper 函数
from .correct_paper import correct_paper, PageNotFoundError

__all__ = ["correct_paper", "PageNotFoundError"]
