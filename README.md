# Pincelate

By [Allison Parrish](https://www.decontextualize.com/)

Pincelate is a machine learning model for spelling English words and sounding
them out, plus a Python module that makes it super simple to do fun and useful
things with the model.

A quick example:

    >>> from pincelate import Pincelate
    >>> pin = Pincelate()
    >>> pin.soundout("pincelate")
    ['P', 'IH1', 'N', 'S', 'AH0', 'L', 'EY1', 'T']
    >>> pin.spell(["HH", "EH", "L", "OW"])
    'hello'

Please see [the documentation](https://pincelate.readthedocs.io/en/latest/) for
more information!

I also did [a series of tutorials on Pincelate for PyCon
2020](https://github.com/aparrish/nonsense-verse-pycon-2020).

## Installation

Machine learning moves fast and breaks things, including backwards
compatibility with models like this. So installation is a bit tricky. You'll
need to install *particular versions* of Tensorflow and Keras to get Pincelate
to work. (For this reason, I highly recommend installing Pincelate in a virtual
environment or conda environment.)

This should do the trick:

    pip install tensorflow==1.15.0 keras==2.2.5 "h5py<3.0.0"

After you've done this, you can install Pincelate:

    pip install pincelate

This will install the code and the pre-trained model.

Pincelate requires *Python 3.6* or later. (It might work on other versions, but
I haven't tested it.) As of this writing, Python 3.8 and later are not yet
supported (because of incompatibilities in some of the required libraries).

## Model card

Following the schema suggested in [Mitchell et
al.](https://arxiv.org/abs/1810.03993).

### Model details

Pincelate was developed and trained by Allison Parrish over the course of 2019.
The current version of the model is 0.0.1. The model consists of a pair of
sequence-to-sequence recurrent neural networks that predict phonetic features
from orthography, and orthography from phonetic features.

The model and accompanying code are available under an MIT open source license.
See `LICENSE` in this repository.

If you make use of Pincelate in your research, please [cite this
repository](https://doi.org/10.5281/zenodo.4726188).

Send questions or comments to [`<aparrish@nyu.edu>`](mailto:aparrish@nyu.edu).

### Intended use

The model has several primary intended uses:

* Guess phonetic pronunciations of English words based on their spelling
  (grapheme to phoneme translation)
* Guess English spellings of arbitrary phonetic pronunciations (phoneme to
  grapheme translation)
* Provide a vector representation of an English word's phonetics,
  based on its spelling
* Facilitate "transformations" of phonetics and spelling through manipulation
  of the model's internal state; e.g., "tinting" phonetics by adjusting the
  logits of the grapheme-to-phoneme model, "blending" words by averaging vector
  representations recovered from RNN hidden states, etc.

The model is designed to facilitate ease of use and creative tinkering, not
fidelity or accuracy. For this reason (and others, outlined below), use of the
model in situations where mistakes in automatically generated phonetic
transcriptions or spellings could lead to critical breakdowns in communication
(like text-to-speech synthesis) is not recommended.

The envisioned users are creative coders interested in using natural language
processing as part of creative writing projects (e.g., computer-generated
poetry).

### Factors and metrics

This section is incomplete! At present, the model training process reports
simple cross-entropy loss and accuracy scores on a validation dataset. A few
goals for future versions of code implementing the training process:

* Test against a dataset entirely held out of the training process
* Test against a dataset of neologisms and non-standard English spellings not
  present in the CMU pronouncing dictionary
* Separately evaluate words in categories pertaining to their semantics,
  etymologies and sociolinguistic use patterns, especially those pertaining to
  groups vulnerable to prejudicial treatment

### Evaluation data and training data

The model is trained and evaluated on data from the [CMU Pronouncing
Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) ("CMUdict"), a freely
available computer-readable phonetic dictionary of English words. At training
time, the data undergo a train/validation split, with membership and
proportion determined at training time (using a user-selectable random seed).

Internally, the model converts CMUdict phones to "phonetic features"; those
features are automatically assigned to each phone based on [a scheme proposed
by
Kirschenbaum](https://web.archive.org/web/20160304092234/http://www.kirshenbaum.net/IPA/ascii-ipa.pdf).

### Ethical considerations and caveats

The spellings and pronunciations in CMUdict reflect a dialect of English named
in the dataset's documentation as "North American English," an equivalent to
"Standard American English." As the model is trained on this dataset, the
model's outputs can be expected to reproduce the phonetics and spelling
conventions of this variety of English. The model does not attempt to
accurately model other accents of English, or to model conventional methods of
spelling those accents.

Orthographic variation, and phonetic variation that surfaces as orthographic
variation, is not value neutral. In particular, ["eye
dialect"](https://en.wikipedia.org/wiki/Eye_dialect) (the deliberate
use of nonstandard spelling to draw attention to a word's pronunciation) can be
used to mock and disparage speakers with particular accents or speech
disorders. The model itself is not trained on any "eye dialect" spellings, but
can be made to produce spellings that resemble eye dialect through careful
manipulation of the model's internal state and decoding process. Applications
making use of this model should take care to limit this affordance.

