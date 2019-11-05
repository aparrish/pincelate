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

