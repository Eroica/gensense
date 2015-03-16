gensense
========

`gensense` is a current university project for the lecture *Formal Semantics* in Computational Linguistics at Heidelberg University during the winter semester of 2014/2015. This project is developed by **Darmin Spahic** and [Sebastian Spaar](https://github.com/Eroica).

Its goal is to assign "senses" to sentences, and thus be able to compare whether two given sentences have a similar meaning. Taking these two example sentences:

1.  *"I fed the dog."*
2.  *"I fed the cat."*

These two sentences' meanings are quite different, even though only a single word has been interchanged. Contrarily, *"The dog was fed by me."* is similar to sentence 1, but this time is using a different word order.

Humans can easily understand the similarity between sentence 1 and its passive form---computers, on the other hand, not so much. We try to implement some basic methods using the Python package `gensim` that allow us to extract the sense of a given sentence and compare them to others.

This project is written in Python 2.7.9 and mainly uses `gensim`'s `word2vec` module (we can't use Python 3 since `gensim` seems to have some incompatibilities with Python 3.4.3).

Tutorial
========

Let's say we want to analyze the grade of similarity between these 3 sentences:

1.  *"I fed the dog."*
2.  *"I fed the cat."*
3.  *"The dog was fed by me."*

In sentence 1 and 2, the *scene* is the same, but the object is different (a different animal is fed, but the "action" is the same). However, one could argue that sentence 3 is a little bit closer in meaning to sentence 1, because it's the same scene and the same object---it is only expressed differently (by using a passive construction).

To calculate a vector that represent a full sentence's meaning, most current implementations do something like this: Look at each individual word, look up that word's vector, and add them up into a sentence vector. That is, if `s` is made up of `w_1`, `w_2`, etc.:

    s = f(w_1, w_2, ..., w_n) s.t.
    s = w_1 + w_2 + ... + w_n

Let's see what `gensim` would calculate for those 3 sentences' similarities. For this task, we used `gensim` to train a model on the *text8* corpus. Keep in mind that the quality of word vectors depends heavily on the corpus used, and *text8* is a relatively small corpus.

    >>> import gensim
    >>> text8 = gensim.models.word2vec.Text8Corpus("text8")
    >>> model = gensim.models.word2vec.Word2Vec(text8, workers=4)
    >>> model.n_similarity("i fed the dog".split(), "i fed the cat".split())
    0.85966112715874687
    >>> model.n_similarity("i fed the dog".split(), "the dog was fed by me".split())
    0.77805009679743709

Our `gensim` model calculates a closer similarity between sentence 1 and 2 than between sentence 1 and 3. This is because our model uses the equation from above to calculate the sentence vector---simply adding every word vector up. In this case, the fact that sentence 3 consists of more words than sentence 1 already manipulates the resulting sentence vector, even though the senses are very similar to each other.

Another possibilty of calculating the sentence vectors is weighting each word differently, so that nouns, verbs, etc., contribute more to the sentence vectors than, say, stop words. For this reason, you may think it is clever to remove all stop words from a sentence (and we also implemented this functionality into `gensense`), however, have a closer look at sentence 3 again:

    The dog was fed by me.

Most of this sentence's meaning is represented by the order of those "unimportant" stop words. If you remove all stop words, the resulting sentence might look like this:

    (The) dog (was) fed (by) me. ---> dog fed me.

Other than (hilariously) changing the original meaning of that sentence, it is still problematic to feed these two sentences (stop words removed) into an additive model like the one used before:

    >>> model.n_similarity("i fed dog".split(), "dog fed me".split())
    0.85002439166352906
    >>> model.n_similarity("i fed dog".split(), "i fed cat".split())
    0.83910888572432696

We are a little bit closer to our goal that sentence 1 and sentence 3 have a closer similarity value with each other, but replace every instance of "dog" with "bird" and the problem still persists:

    >>> model.n_similarity("i fed bird".split(), "i fed cat".split())
    0.81879908349895825
    >>> model.n_similarity("i fed bird".split(), "bird fed me".split())
    0.81658789366759033

(Also keep in mind that we use the same, simple model every time, namely *text8*, and the quality of the word vectors change depending on the model. On top of that, all of those sentences are fairly short.)

This is where `gensense` comes in. In `gensense`, every sentence is analyzed and clustered into groups of similar meanings. The clustering algorithm is an implementation of the [*Chinese restaurant process*](http://en.wikipedia.org/wiki/Chinese_restaurant_process). This way, we are looking to reduce a sentence's meaning to those words that play the most important role in a sentence. In case of sentence 1, this might be that "a dog is being fed", hence, `dog + fed`. For sentence 2 it is `cat + fed`. For sentence 3---since the action taking place is the same as in sentence 1---it is also `dog + fed`! While the rest of words (*by*, *me*, etc.) end up in another cluster.

    >>> import gensense
    >>> dog = gensense.sentence.Sentence("i fed the dog")
    >>> dog_passive = gensense.sentence.Sentence("the dog was fed by me")

Let's see what clusters have been determined for these 2 sentences:

    >>> dog.clusters
    [['i', 'the'], ['fed', 'dog']]
    >>> dog_passive.clusters
    [['the', 'was'], ['dog', 'fed'], ['by'], ['me']]

We can see that our algorithm put *dog* and *fed* into the same cluster, and the other words in different ones. This way, comparing the meaning of sentence 1 and sentence 3 can be reduced to comparing the similarity of these clusters, which are of course the same:

    >>> model.n_similarity("fed dog".split(), "dog fed".split())
    1.0000000000000002

QED.

Caveats
-------

Due to the nature of the *Chinese restaurant process*, there is a certain amount of randomness involved in creating the clusters, especially for short sentences like sentence 1, 2 and 3. You can re-roll the dies by calling the `clusterize()` method:

    >>> dog.clusterize()
    >>> dog.clusters
    [['i', 'the'], ['fed'], ['dog']]

As stated before, for short sentences this process is a little bit more on the random side. Compare this with a longer sentence (retrieved from [nytimes.com](http://www.nytimes.com/) on March 14th, 2015):

    >>> text = "decades after dan jones last flight over vietnam ended in wreckage and blood his effort to see that his fallen marine crew members be honored is reaching its end"
    >>> marine = gensense.sentence.Sentence(text)
    >>> marine.clusters
    [['decades', 'after', 'last', 'over', 'vietnam', 'ended', 'wreckage', 'his', 'effort', 'reaching', 'end'], ['dan', 'jones', 'members'], ['flight', 'marine', 'crew'], ['in', 'and', 'to', 'see', 'that', 'be', 'is'], ['blood'], ['fallen', 'its'], ['honored']]
    >>> marine.clusterize()
    >>> marine.clusters
    [['decades', 'after', 'flight', 'vietnam', 'ended', 'effort', 'marine', 'crew', 'reaching', 'end'], ['dan', 'jones'], ['last', 'his', 'fallen', 'honored'], ['over', 'in', 'and', 'to', 'see', 'that', 'members', 'be', 'is'], ['wreckage', 'blood'], ['its']]


Future work
-----------

The `clusterize()` algorithm groups similar words into clusters, and adds those word vectors up into a *cluster vector*. While the *cluster vectors* can be calculated using any function that calculates sentence vectors (look into the API documentation below), right now, there is no possibility of weighting each cluster differently.


API
===

The `gensim` package is split into 3 parts: `gensim.sentence`, `gensim.corpus` and `gensim.evaluation`.

`gensim.sentence`
-----------------

This module implements the `Sentence` object which wraps a sentence string and every vector of that sentence's words. A `word2vec` model supplies the word vectors. By default, `gensense` uses `gensense.sentence.MODEL` which is a model trained by `gensim` on the *text8* corpus.

    >>> import gensense
    >>> dog = gensense.sentence.Sentence("i fed the dog")

If you have an own model trained by `word2vec`/`gensim`, pass it as a 2nd argument:

    >>> dog = gensense.sentence.Sentence("i fed the dog", my_model)

Internally, every `Sentence` object is an `OrderedDict` which means you can access every word vector by using `Sentence["key"]`:

    >>> dog["fed"]
    array([ -2.32517533e-02,   1.27635270e-01,   1.52028486e-01,
         6.63337559e-02,  -1.57799542e-01,  -3.17575276e-01,
         7.05860704e-02,   4.00045291e-02,  -1.94212601e-01,
        ...
        ], dtype=float32)

For every `Sentence` object, the `*` and `+` is overloaded to allow a simple weighting of individual words.

Other `Sentence` methods:

*   `removeStopWords(self, stop_words=STOP_WORDS):`: Iterates over the sentence and removes all words that are found in `stop_words`. If no list is supplied, a default list of stop words from NLTK is used.
*   `clusterize(self)`: The implementation of the [*Chinese restaurant process*](http://en.wikipedia.org/wiki/Chinese_restaurant_process). The sentence is grouped into smaller clusters of similar words. Each cluster has an independent *cluster vector* that can be used to calculate sentence similarities. This method creates the following fields:
    *   `self.clusters`: The list of clusters, each cluster is a list of words.
    *   `self.cluster_sums`: The list of cluster vectors.


`gensim.corpus`
---------------

This module is used to access Mitchell's & Lapata's evaluation corpus that is available [here](TODO). For projects involving sentence similarities, this corpus is useful to have a comparison to one's own similarity calculations.

The corpus is initialized as an `EvaluationCorpus` object. When being initialized, it iterates over the text file, which looks something like this:

    participant1 verbobjects 2 knowledge use influence exercise 5
    participant1 verbobjects 2 war fight battle win 5
    participant1 verbobjects 2 support offer help provide 7
    participant1 verbobjects 2 technique develop power use 2

Every participant is put into `EvaluationCorpus` (a dictionary). Then, each line is parsed for the word pairs and the similarity grade, for instance:

    participant1 verbobjects 2 knowledge use influence exercise 5
    ---> EvaluationCorpus["participant1"]: [["knowledge use", "influence exercise", 5]]

Internally, each word pair is also transformed in a `Sentence` object as described in `gensense.sentence`.

`gensim.evaluate`
-----------------

This module provides functions to compare sentences with each other, and a way to test these similarity functions on Mitchell's and Lapata's corpus.

The three most interesting functions here are probably `similarity`, `compare`, and `evaluate_ml_corpus`:

*   `similarity(sentence1, sentence2, sv_function=sv_add)`: Takes two sentences, and calculates their similarity using the function provided (last argument). Any function that is able to calculate a sentence vector on a `Sentence` object (for instance, by iterating over `Sentence.values()`) can be used. By default, the sentence vector is calculated by simple addition of every word vector (`sv_add`), but we also provide three other functions:
    *   (`sv_add(sentence)`: Calculate sentence vector by adding each word vector up.)
    *   `sv_weightadd(sentence, weights=WEIGHTS)`: Also adds each vector up into a sentence vector, but multiplies each vector by a scalar depending on their POS tag. These weights can be customized by providing a dictionary in the following format:

            weights = { 'NN': 0.9, 'JJ': 0.5, ... }

        Sample standard weights are in `gensense.evaluation.WEIGHTS`. It is best to use a `defaultdict` for this weight dictionary.

    *   `sv_multiply(sentence)`: Creates a sentence vector by multiplying each i-th vector component of one word with each i-th component of the other word vectors.
    *   `sv_kintsch(sentence, model=MODEL)`: Looks for the predicate in the sentence, and looks up the closest word of that predicate in `model`. `model` should be a model generated by `gensim` or `word2vec`. By default, the standard model in `gensense.sentence.MODEL` is used.

*  `compare(sentence1, sentence2, sv_function=sv_add)`: Similar to the function above (and might be merged in the future), but takes into account the word clusters of every sentence. Right now, this function asks you to choose the most relevant cluster, but this can be automated by training a neural network on it.

*   `evaluate_ml_corpus(corpus=ML_CORPUS, participant="participant1")`: This function calculates the four similarity function (described above) on Mitchell's and Lapata's evaluation corpus. By default, the sentences as seen by *participant 1* is  used for calculation, but any other participant (of the 162) can be used by overriding `participant=`.

    Sample usage:

        >>> gensense.evaluation.evaluate_ml_corpus(participant="participant162")
        1st Sentence          2nd Sentence            M&L    +    W.+    *    K.
        --------------------  --------------------  -----  ---  -----  ---  ----
        black hair            right hand                1    2      1    2     2
        high price            low cost                  1    6      5    6     6
        social event          special circumstance      1    2      1    3     2
        small house           important part            1    2      1    2     2
        large number          vast amount               6    5      3    4     5
        certain circumstance  economic condition        2    3      1    4     3
        old person            elderly lady              4    3      1    3     3
        little room           similar result            1    2      1    1     1
        earlier work          early stage               5    4      3    4     4
        practical difficulty  cold air                  1    1      1    1     1
        new law               modern language           1    2      1    2     2
        new body              significant role          1    2      1    2     2
        previous day          long period               1    3      1    3     3
        new technology        public building           2    3      2    3     3
        social activity       whole system              1    2      1    2     2
        general principle     basic rule                4    4      2    3     4
        northern region       industrial area           1    3      2    4     3
        new information       general level             1    2      2    2     2
        dark eye              left arm                  1    2      1    2     2
        high point            particular case           1    3      1    3     3

        1st Sentence       2nd Sentence            M&L    +    W.+    *    K.
        -----------------  --------------------  -----  ---  -----  ---  ----
        short time         rural community           1    1      1    1     1
        different kind     various form              6    4      3    3     4
        hot weather        further evidence          1    1      1    1     1
        central authority  local office              1    3      2    2     3
        economic problem   new situation             1    3      1    4     3
        effective way      efficient use             5    5      2    4     5
        new life           early age                 5    3      1    3     3
        political action   economic development      3    4      2    3     4
        european state     present position          1    3      2    3     3
        early evening      good effect               1    1      1    1     2
        major issue        american country          1    2      1    1     2
        older man          elderly woman             2    4      3    5     4
        better job         good place                3    4      2    3     4
        federal assembly   national government       6    5      4    5     5
        whole country      different part            1    3      1    4     3
        large quantity     great majority            5    3      1    2     3


Appendix
--------

*   *Mitchell & Lapata 2008: Vector-based Models of Semantic Composition*: http://www.aclweb.org/anthology/P08-1028
