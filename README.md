gensense
========

`gensense` is a current university project for the lecture *Formal Semantics* in Computational Linguistics at Heidelberg University during the winter semester of 2014/2015. This project is developed by **Darmin Spahic** and [Sebastian Spaar](https://github.com/Eroica).

Its goal is to assign "senses" to sentences, and thus be able to compare whether two given sentences have a similar meaning. Taking these two example sentences:

1.  *"I was feeding the dog."*
2.  *"I was feeding the cat."*

These two sentences' meanings are quite different, even though only a single word has been interchanged. Contrarily, *"The dog was fed by me."* is similar to sentence 1, but this time is using a different word order.

Humans can easily understand the similarity between sentence 1 and its passive form---computers, on the other hand, not so much. We try to implement some basic methods using the Python package `gensim` that allow us to extract the sense of a given sentence and compare them to others.

Development Information and API
-------------------------------

This project is written in Python 2.7.9 and mainly uses `gensim`'s `word2vec` module (we can't use Python 3 since `gensim` seems to have some incompatibilities with Python 3.4.3).

This project is still on-going. Stay tuned for updates and usage instructions.

Appendix
--------

*   *Michell & Lapata 2008: Vector-based Models of Semantic Composition*: http://www.aclweb.org/anthology/P08-1028
