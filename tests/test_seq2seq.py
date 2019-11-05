import unittest
import pincelate
import pincelate.cmudictdata


class TestSeq2Seq(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("loading data")
        cls.data = pincelate.cmudictdata.load()
        print("done")
        cls.tts = pincelate.cmudictdata.tts(cls.data)

    def test_train(self):
        from pincelate.seq2seq import Seq2Seq
        testmodel = Seq2Seq(
                src_vocab=self.data['orthography_vocab'],
                target_vocab=self.data['phoneme_feature_vocab'],
                activation='sigmoid',
                loss='binary_crossentropy',
                hps=dict(
                    optimizer='adam',
                    lr=0.001,
                    bidi=True,
                    enc_rnn_units=4,
                    dec_rnn_units=4,
                    enc_rnn_dropout=0.2,
                    dec_rnn_dropout=0.2,
                    vae=False
                )
        )
        testmodel.make_models()
        testmodel.compile_models()
        print(testmodel.training_model.summary())
        history = testmodel.fit(
                [self.tts['orth_data_train'][:1000],
                    self.tts['phon_data_train'][:1000]],
                self.tts['phon_target_train'][:1000],
                batch_size=256,
                epochs=1,
                validation_data=(
                    [self.tts['orth_data_valid'][:100],
                        self.tts['phon_data_valid'][:100]],
                    self.tts['phon_target_valid'][:100]))
        testmodel.save("tests/testmodel")
        loaded = Seq2Seq.load("tests/testmodel")

        # ensure that histories are identical
        self.assertEqual(loaded.training_history, history.history)

        # all fields should be identical
        for field in Seq2Seq.SERIALIZE_FIELDS:
            self.assertEqual(
                    getattr(loaded, field),
                    getattr(testmodel, field))


if __name__ == '__main__':
    unittest.main()
