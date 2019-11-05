from pincelate.seq2seq import Seq2Seq
from pincelate.featurephone import phone_feature_map
from pincelate.seq2seq import softmax_temperature, sigmoid_top_n

import numpy as np

__author__ = 'Allison Parrish'
__email__ = 'allison@decontextualize.com'
__version__ = '0.0.1'


class Pincelate:
    """

    Loads and provides an API for sequence-to-sequence models for spelling
    and sounding out.

    """

    def __init__(self, model_path_prefixes=()):
        """Initialize Pincelate model.

        By default, pretrained models will be loaded from package data.
        If you train your own custom models with ``pincelate.train``, you can
        load those models using the ``model_path_prefixes`` argument.

        Parameters
        ----------

        model_path_prefixes : tuple, optional
            Prefix of (orth2phon, phon2orth) custom models to load.

        """

        # if custom model paths were provided, load them
        if len(model_path_prefixes) > 0:
            self.orth2phon = Seq2Seq.load(model_path_prefixes[0])
            self.phon2orth = Seq2Seq.load(model_path_prefixes[1])
        # otherwise, use the models in the package
        else:
            self.orth2phon = Seq2Seq.load_from_package(
                    'models/orth-phon-enc256-dec256')
            self.phon2orth = Seq2Seq.load_from_package(
                    'models/phon-orth-enc256-dec256')

        # looking up the closest arpabet phonemes when sounding out
        self.targets = []

        # add entries for both stressed and unstressed vowels
        for phone, feats in [(k, v) for k, v in phone_feature_map.items()
                             if 'vwl' in v]:
            unstr_vector = self.orth2phon.vectorize_target([feats], 1)[0, 0]
            str_vector = self.orth2phon.vectorize_target([feats], 1)[0, 0]
            str_vector[self.orth2phon.target_vocab_idx_map["str"]] = 1.
            self.targets.append((phone+"0", unstr_vector))
            self.targets.append((phone+"1", str_vector))

        # consonants can't be stressed
        self.targets.extend(
                [(k, self.orth2phon.vectorize_target([v], 1)[0, 0])
                 for k, v in phone_feature_map.items() if 'vwl' not in v]
        )

    def soundout(self, s):
        """'Sounds out' the string, returning a list of Arpabet phonemes.

        Parameters
        ----------
        s : str
            The string to sound out. Should contain only lowercase ASCII
            characters.

        Returns
        -------
        list
            List of Arpabet phonemes

        Examples
        --------

        >>> from pincelate import Pincelate
        >>> pin = Pincelate()
        >>> pin.soundout("hello")
        ['HH', 'EH1', 'L', 'OW0']

        """

        state_val = self.orth2phon.infer(
                self.orth2phon.vectorize_src("^"+s+"$", maxlen=31))
        target_seq = self.orth2phon.vectorize_target([["beg"]], 1)
        decoding = self.orth2phon.decode_sequence(state_val, target_seq)
        seq = [self.closest(item[0, 0]) for item in decoding]
        return seq[seq.index("^")+1:seq.index("$")]

    def spell(self, phones, temperature=0.25):
        """Produces a plausible spelling for a list of Arpabet phonemes.

        Parameters
        ----------
        phones : list
            A list of Arpabet phonemes. Vowels may optionally have stress
            numbers appended to the end (i.e., you can provide ``EH``,
            ``EH0``, ``EH1``, ``EH2``).
        temperature : float
            Temperature for softmax sampling. Larger values will yield
            more unusual results.

        Returns
        -------
        str
            A spelling of the provided phonemes

        Examples
        --------

        >>> from pincelate import Pincelate
        >>> pin = Pincelate()
        >>> pin.spell(['HH', 'EH1', 'L', 'OW0'])
        'hello'

        """

        feats = []
        for item in phones:
            # account for stressed phonemes
            if item[-1] in ('1', '2'):
                this_feat = phone_feature_map[item[:-1]]
                this_feat = this_feat + ("str",)
            elif item[-1] == '0':
                this_feat = phone_feature_map[item[:-1]]
            else:
                this_feat = phone_feature_map[item]
            feats.append(this_feat)
        state_val = self.phon2orth.infer(
            self.phon2orth.vectorize_src(
                [["beg"]] + feats + [["end"]], maxlen=31)
        )
        target_seq = self.phon2orth.vectorize_target("^", 1)
        decoding = self.phon2orth.decode_sequence(
            state_val,
            target_seq,
            transform=lambda x: softmax_temperature(x, temperature))
        seq = [self.phon2orth.to_target_vocab(item, 1)[0][0]
               for item in decoding]
        return ''.join(seq[seq.index("^")+1:seq.index("$")])

    def phonemefeatures(self, s):
        """Produces an array of phoneme feature probabilities for string.

        This function operates like `soundout`, except it omits the
        nearest-neighbor phoneme lookup at the end, returning the raw phoneme
        feature probabilities instead.

        You can "spell" this array of phoneme features using the
        ``spellfeatures()`` method.

        Parameters
        ----------
        s : str
            The string to sound out. Should contain only lowercase ASCII
            characters.

        Returns
        -------
        numpy.array
            A numpy array of shape (*n*, *m*) where *n* is the number of
            predicted phonemes (including begin/end tokens) and *m* is the
            number of phoneme features in the training data (32 for the
            included pretrained model)

        Examples
        --------

        >>> from pincelate import Pincelate
        >>> pin = Pincelate()
        >>> feats = pin.phonemefeatures("hello")
        >>> feats.shape
        (6, 32)

        """

        phon_state_val = self.orth2phon.infer(
            self.orth2phon.vectorize_src("^"+s+"$", maxlen=31))
        phon_target_seq = self.orth2phon.vectorize_target([["beg"]], 1)
        phon_decoding = self.orth2phon.decode_sequence(
            phon_state_val,
            phon_target_seq
        )
        phon_seq = []
        for item in phon_decoding:
            phon_seq.append(item[0, 0])
            # if the 'end' token is in the top three features, stop
            if 'end' in self.orth2phon.to_target_vocab(
                    sigmoid_top_n(item, 3), maxlen=1)[0]:
                break
        phon_seq = np.array(phon_seq)
        return phon_seq

    def phonemestate(self, s):
        """Calculates hidden state of the spelling model's decoder for string.

        This hidden state can be used for various purposes, including as a
        representation of the way the string sounds (for the purpose of
        similarity searches).

        You can decode a spelling from a state returned from this method
        with the ``spellstate()`` method.

        Parameters
        ----------
        s : str
            The string to sound out. Should contain only lowercase ASCII
            characters.

        Returns
        -------
        numpy.array
            Array of shape (*n*,), where *n* is the number of dimensions in the
            spelling model's hidden state (256 for the included pretrained
            model)


        Examples
        --------

        The following shows how to Calculate and compare the distance between
        the sound of two pairs of words.

        >>> from pincelate import Pincelate
        >>> from numpy.linalg import norm
        >>> pin = Pincelate()
        >>> bug2rug = norm(pin.phonemestate("bug") - pin.phonemestate("rug"))
        >>> bug2zap = norm(pin.phonemestate("bug") - pin.phonemestate("zap"))
        >>> bug2rug < bug2zap
        True

        """

        state_val = self.orth2phon.infer(
                self.orth2phon.vectorize_src("^"+s+"$", maxlen=31))
        target_seq = self.orth2phon.vectorize_target([["beg"]], 1)
        decoding = self.orth2phon.decode_sequence(state_val, target_seq)
        seq = []
        for item in decoding:
            seq.append(item[0, 0])
            # if the 'end' token is in the top three features, stop
            if 'end' in self.orth2phon.to_target_vocab(
                    sigmoid_top_n(item, 3), maxlen=1)[0]:
                break
        phon_seq = np.array(seq)
        orth_state_val = self.phon2orth.infer(np.array([phon_seq]))
        return orth_state_val.squeeze()

    def spellfeatures(self, vec, temperature=0.25):
        """Produces a plausible spelling of an array of phoneme features.

        Arrays of phoneme features are returned from the ``phonemefeatures()``
        and ``vectorizefeatures()`` methods.

        Parameters
        ----------
        vec : numpy.array
            A numpy array of shape (*n*, *m*), where *n* is the number of
            phonemes in the word (including begin/end tokens) and *m*
            is the number of phoneme features in the training data (32 for the
            included pretrained model)
        temperature : float
            Temperature for softmax sampling. Larger values will yield
            more unusual results.

        Returns
        -------
        str
            A spelling of the provided phoneme features

        Examples
        --------

        >>> from pincelate import Pincelate
        >>> pin = Pincelate()
        >>> bee = pin.vectorizefeatures([
        ...     ['beg'], ['blb', 'stp', 'vcd'], ['hgh', 'fnt', 'vwl'], ['end']
        ... ])
        >>> pin.spellfeatures(bee)
        'bee'

        """

        state_val = self.phon2orth.infer(np.expand_dims(vec, axis=0))
        target_seq = self.phon2orth.vectorize_target("^", 1)
        decoding = self.phon2orth.decode_sequence(
            state_val,
            target_seq,
            transform=lambda x: softmax_temperature(x, temperature))
        seq = [self.phon2orth.to_target_vocab(item, 1)[0][0]
               for item in decoding]
        return ''.join(seq[seq.index("^")+1:seq.index("$")])

    def spellstate(self, state, temperature=0.25):
        """Produces a plausible spelling from spelling model's hidden state.

        Parameters
        ----------
        state : numpy.array
            Array of shape (*n*,), where *n* is the number of dimensions in the
            spelling model's hidden state (256 for the included pretrained
            model)
        temperature : float
            Temperature for softmax sampling. Larger values will yield
            more unusual results.

        Returns
        -------
        str
            A spelling of the provided phoneme state

        Examples
        --------

        >>> ai = (pin.phonemestate("artificial"),
        ...      pin.phonemestate("intelligence"))
        >>> pin.spellstate((ai[0] + ai[1]) / 2)
        'intelifical'

        """

        orth_state_seq = self.phon2orth.vectorize_target("^", 1)
        orth_decoding = self.phon2orth.decode_sequence(
            np.expand_dims(state, axis=0),
            orth_state_seq,
            transform=lambda x: softmax_temperature(x, temperature)
        )
        orth_seq = [self.phon2orth.to_target_vocab(item, 1)[0][0]
                    for item in orth_decoding]

        # trim start/end tokens
        try:
            start = orth_seq.index("^")+1
        except ValueError:
            start = 0
        try:
            end = orth_seq.index("$")
        except ValueError:
            end = -1
        return ''.join(orth_seq[start:end])

    def manipulate(self, s, letters=None, features=None, temperature=0.25):
        """Manipulate a round-trip respelling of a string

        This method 're-spells' words by first translating them to phonetic
        features then back to orthography. The provided values are used to
        attenuate or emphasize the probability of the given letters and
        phonetic features at each step of the spelling and sounding out
        process. Specifically, the decoded probability of the given item
        (letter or feature) is raised to the power of ``np.exp(n)``,
        where ``n`` is the provided value. A value of 0 will affect no change;
        negative values will increase the probability, positive values will
        decrease the probability. (A good range to try out is -10 to +10.)

        Parameters
        ----------
        s : str
            String to be re-spelled. Should contain only ASCII lowercase
            characters.
        letters : dict
            Dictionary mapping letters to exponent values
        features : dict
            Dictionary mapping phonetic features to exponent values
        temperature : float
            Temperature for softmax sampling. Larger values will yield
            more unusual results.

        Returns
        -------
        str
            A re-spelling of the provided word

        Examples
        --------

        Respell without using particular letters:

        >>> from pincelate import Pincelate
        >>> pin = Pincelate()
        >>> pin.manipulate("cheese", letters={'e': 10})
        "chi's"

        Respell emphasizing certain phonetic features:

        >>> from pincelate import Pincelate
        >>> pin = Pincelate()
        >>> pin.manipulate("nod", features={'alv': 10, 'blb': -10})
        'mob'

        Produce a less plausible spelling:

        >>> from pincelate import Pincelate
        >>> pin = Pincelate()
        >>> [pin.manipulate("alphabet", temperature=1.5) for i in range(5)]
        ['alphabet', 'alphabey', 'alphibete', 'alfabet', 'alphabette']

        """

        if letters is None:
            letters = {}
        if features is None:
            features = {}

        # split out parameters
        orth_exps = self._zero_map(
                self.orth2phon.src_vocab_idx_map, **letters)

        phon_exps = self._zero_map(
                self.phon2orth.src_vocab_idx_map, **features)

        # translate from orthography to phonetic features
        phon_state_val = self.orth2phon.infer(
            self.orth2phon.vectorize_src("^"+s+"$", maxlen=31))
        phon_target_seq = self.orth2phon.vectorize_target([["beg"]], 1)
        phon_decoding = self.orth2phon.decode_sequence(
            phon_state_val,
            phon_target_seq,
            # inference modification happens here
            transform=lambda x: x ** np.exp(phon_exps)
        )
        phon_seq = []
        for item in phon_decoding:
            phon_seq.append(item[0, 0])
            # if the 'end' token is in the top three features, stop
            if 'end' in self.orth2phon.to_target_vocab(
                    sigmoid_top_n(item, 3), maxlen=1)[0]:
                break
        phon_seq = np.array(phon_seq)

        # translate from phonetic features to orthography
        orth_state_val = self.phon2orth.infer(np.array([phon_seq]))
        orth_state_seq = self.phon2orth.vectorize_target("^", 1)
        orth_decoding = self.phon2orth.decode_sequence(
            orth_state_val,
            orth_state_seq,
            # orthography inference manipulation
            transform=lambda x: softmax_temperature(
                x ** np.exp(orth_exps), temperature)
        )
        orth_seq = [self.phon2orth.to_target_vocab(item, 1)[0][0]
                    for item in orth_decoding]

        # trim start/end tokens
        try:
            start = orth_seq.index("^")+1
        except ValueError:
            start = 0
        try:
            end = orth_seq.index("$")
        except ValueError:
            end = -1
        return ''.join(orth_seq[start:end])

    def closest(self, vec):
        "Finds the closest Arpabet phoneme for the given feature vector"
        return sorted(
                self.targets, key=lambda x: np.linalg.norm(vec - x[1]))[0][0]

    def vectorizefeatures(self, arr):
        """Vectorizes a list of lists of phoneme features.

        Helpful if you want to author phoneme features "by hand," instead of
        (e.g.) using `phonemefeatures` to infer them from spelling.

        Parameters
        ----------
        arr : list of lists
            List of list of phoneme features (see `pincelate.featurephone`
            for a list)

        Returns
        -------
        numpy.array
            A numpy array of shape (*n*, *m*), where *n* is the number of
            phonemes and *m* is the number of phoneme features.

        Examples
        --------

        >>> from pincelate import Pincelate
        >>> pin = Pincelate()
        >>> bee = pin.vectorizefeatures([
        ...     ['beg'], ['blb', 'stp', 'vcd'], ['hgh', 'fnt', 'vwl'], ['end']
        ... ])
        >>> pin.spellfeatures(bee)
        'bee'

        """

        return self.orth2phon.vectorize_target(arr, maxlen=31)[0]

    def featureidx(self, feat):
        """Index of a feature in the phoneme feature vocabulary.

        Examples
        --------

        >>> from pincelate import Pincelate
        >>> pin = Pincelate()
        >>> pug = pin.phonemefeatures("pug")
        >>> pug[1][pin.featureidx('vcd')] = 1
        >>> pin.spellfeatures(pug)
        'bug'

        """

        return self.orth2phon.target_vocab_idx_map[feat]

    def _zero_map(self, vocab_map, **t):
        "Creates a numpy array with specified values from given vocabulary"
        vals = np.zeros(len(vocab_map))
        for k, v in t.items():
            vals[vocab_map[k]] = v
        return vals
