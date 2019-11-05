import unittest
import numpy as np
import os
from os.path import join as opj

from pincelate import Pincelate

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class TestPincelate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pin = Pincelate()

    def test_load_custom_models(self):
        from pincelate import Pincelate
        pin = Pincelate(
                ('pincelate/models/orth-phon-enc256-dec256',
                 'pincelate/models/phon-orth-enc256-dec256'))
        self.assertEqual(pin.soundout("hello"), ['HH', 'EH1', 'L', 'OW0'])

    def test_soundout(self):
        self.assertEqual(self.pin.soundout("hello"), ['HH', 'EH1', 'L', 'OW0'])

    def test_spell(self):
        self.assertEqual(self.pin.spell(['HH', 'EH1', 'L', 'OW0']), 'hello')

    def test_phonemefeatures(self):
        with open(opj(TEST_DIR, "allison-phoneme-features.npy"), "rb") as fh:
            arr = np.load(fh)
            self.assertTrue(
                    np.all(self.pin.phonemefeatures("allison") == arr))
            self.assertIn(self.pin.spellfeatures(arr), ("allison", "alison"))

    def test_phonemestate(self):
        with open(opj(TEST_DIR, "allison-phoneme-state.npy"), "rb") as fh:
            arr = np.load(fh)[0]
            self.assertTrue(np.all(self.pin.phonemestate("allison") == arr))
            self.assertIn(self.pin.spellstate(arr), ("allison", "alison"))

    def test_manipulate(self):
        res = self.pin.manipulate("illicit", letters={'i': 10, 'y': -2},
                                  temperature=0.05)
        self.assertGreater(res.count('y'), res.count('i'))
        res = self.pin.manipulate("ticktock", features={'nas': -5, 'vcd': -5},
                                  temperature=0.05)
        self.assertGreaterEqual(res.count('d'), res.count('t'))
        self.assertGreater(res.count('g'), res.count('k'))

    def test_vectorizefeatures(self):
        bee = self.pin.vectorizefeatures([
            ['beg'], ['blb', 'stp', 'vcd'], ['hgh', 'fnt', 'vwl'], ['end']])
        with open(opj(TEST_DIR, "bee-features.npy"), "rb") as fh:
            arr = np.load(fh)
            self.assertTrue(np.all(bee == arr))

    def test_featureidx(self):
        from pincelate.featurephone import phone_feature_map
        self.assertEqual(self.pin.featureidx('beg'),
                         self.pin.orth2phon.target_vocab_idx_map['beg'])


if __name__ == '__main__':
    unittest.main()
