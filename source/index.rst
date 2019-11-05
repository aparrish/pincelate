.. Pincelate documentation master file, created by
   sphinx-quickstart on Mon Nov  4 17:05:31 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pincelate documentation
=======================

Pincelate is a machine learning model for spelling English words and sounding
them out, plus a Python module that makes it super simple to do fun and useful
things with the model.

A quick example::

    >>> from pincelate import Pincelate
    >>> pin = Pincelate()
    >>> pin.soundout("pincelate")
    ['P', 'IH1', 'N', 'S', 'AH0', 'L', 'EY1', 'T']
    >>> pin.spell(["HH", "EH", "L", "OW"])
    'hello'

Pronunciations are specified and returned in a phonetic alphabet called
`Arpabet <https://en.wikipedia.org/wiki/ARPABET>`_. The model is trained on the
`CMU Pronouncing Dictionary <http://www.speech.cs.cmu.edu/cgi-bin/cmudict>`_.
(Note that this module does not itself perform text-to-speech!)

Installation
------------

First, `install Tensorflow <https://www.tensorflow.org/install>`_. (Tensorflow
isn't included in the package dependencies because you need to choose either
the GPU or CPU version.) After that, install with ``pip``::

    pip install pincelate

This will install the code and the pre-trained model.

Documentation
-------------

In addition to the documentation provided below, see `this Jupyter Notebook
<TK>`_ with a tutorial and cookbook.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pincelate
   training


About
-----

English spelling doesn't have much to do with the way English is pronounced. If
you need to write a computer program that incorporates some knowledge of how
words are pronounced, or how particular pronunciations correspond to spelling,
you might use a database like the `CMU Pronouncing Dictionary
<http://www.speech.cs.cmu.edu/cgi-bin/cmudict>`_, which contains more than
100,000 words along with their pronunciations. (I use this database so much
that I made a `Python module <https://pypi.org/project/pronouncing/>`_ to
simplify the process.)

But what about when you want to automatically sound out words that aren't in
the database? Or when you want to automatically spell a word whose
pronunciation isn't in the database? In that case, you need some kind of
*model* that specifies the rules or patterns for spelling and sounding out in
English. In computer science and machine learning, the terminology for the
process of sounding out words is "grapheme to phoneme conversion," and the
process for spelling words based on their pronunciation is "phoneme to grapheme
conversion." (*Phoneme* is a fancy word for "sound," and *grapheme* is a fancy
word for "letter.")

I'm a computer programmer, poet and college professor who often makes work that
deals with how words are spelled and how they're pronounced. I needed an
easy-to-use Python module that does grapheme-to-phoneme conversion and
phoneme-to-grapheme conversion that I can use in my own practice and in my
classes and workshops. So I made Pincelate.

(Why "Pincelate"? An earlier version of the model that used a variational
autoencoder produced this string from the averaged latent vectors of the words
``phonetic``, ``neologism``, ``continuous``, and ``state``. The variational
aspect of the model has been shelved for a bit, but the name stuck!)


The Model
---------

Grapheme to phoneme conversion is `well-worn territory
<https://scholar.google.com/scholar?hl=en&q=grapheme+to+phoneme>`_
for researchers in machine learning and computer science (as is phoneme to
grapheme conversion, though to a lesser extent), and there's nothing especially
interesting from a technical standpoint in my model. But it performs pretty
well in practice!

Pincelate pairs two sequence-to-sequence models: one that translates
orthography (i.e., graphemes) to phonemes, and another that translates phonemes
to orthography. Both are trained on data from the CMU Pronouncing Dictionary.
Orthography is represented as a sequence of one-hot encoded letters. Phonemes,
on the other hand, are represented as a sequence of k-hot encoded *phoneme
features*, meaning that the phoneme /B/, for example, is represented as
``('blb', 'vcd', 'stp')`` (bilabial voiced stop). The goal of using phoneme
features instead of whole phonemes is firstly to improve the overall accuracy
of the model, but it also enables certain techniques (such as
boosting/attenuating certain features in the decoding process) that would be
impossible otherwise.

The source code for the sequence-to-sequence model architecture is provided in
the package (``pincelate/seq2seq.py``). It's implemented in Keras and uses a
GRU-style RNN. (Further details, documentation and evaluation metrics to come.)

License
-------

Copyright (c) 2019 Allison Parrish.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
