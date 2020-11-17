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

