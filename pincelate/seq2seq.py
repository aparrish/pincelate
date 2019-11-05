from types import SimpleNamespace

from keras.models import Model
from keras.layers import Input, GRU, Dense, Masking, Concatenate, \
    Bidirectional
from keras import optimizers
import numpy as np


class Seq2Seq:
    """
    A sequence to sequence model with recurrent neural networks, plus
    some helpful functions for training, saving, loading, inference and
    decoding.
    """
    def __init__(self, src_vocab, target_vocab, activation, loss,
                 training_history=None, hps=None):
        self.src_vocab = src_vocab
        self.src_vocab_idx_map = {k: i for i, k in enumerate(src_vocab)}
        self.target_vocab = target_vocab
        self.target_vocab_idx_map = {k: i for i, k in enumerate(target_vocab)}
        self.activation = activation
        self.loss = loss
        self.training_history = training_history
        if hps is None:
            hps = {}
        if isinstance(hps, dict):
            self.hps = SimpleNamespace(**hps)
        elif isinstance(hps, SimpleNamespace):
            self.hps = hps
        else:
            raise ValueError("incompatible type for hps")

    def make_models(self):

        #
        # training models
        #

        training_encoder_inputs = Input(
            shape=(None, len(self.src_vocab)),
            name='training_encoder_inputs'
        )

        # masking to exclude zero-padding in loss/accuracy calculation
        training_encoder_mask = Masking(
                0.0,
                name='training_encoder_mask')(training_encoder_inputs)

        # two setups for encoder: bidirectional and unidirectional
        if self.hps.bidi:
            training_encoder = Bidirectional(
                    GRU(
                        self.hps.enc_rnn_units,
                        recurrent_dropout=self.hps.enc_rnn_dropout,
                        return_state=True),
                    name='training_encoder_bidi')
            # Bidirectional wrapper returns both forward and backward state
            (_,
             training_encoder_fstate,
             training_encoder_bstate) = training_encoder(training_encoder_mask)
            training_encoder_state = Concatenate(
                    name='training_encoder_bidi_state')(
                            [training_encoder_fstate, training_encoder_bstate])
        else:
            training_encoder = GRU(
                    self.hps.enc_rnn_units,
                    recurrent_dropout=self.hps.enc_rnn_dropout,
                    return_state=True,
                    name='training_encoder')
            _, training_encoder_state = training_encoder(training_encoder_mask)

        # dimensions of encoder hidden state may not match the dimensions of
        # the decoder's hidden state (and the keras implementation doesn't
        # allow us to set these separately). so we need a layer to match the
        # two.
        initial_state = Dense(self.hps.dec_rnn_units, name='initial_state')

        training_initial_state = initial_state(training_encoder_state)

        training_decoder_inputs = Input(
                shape=(None, len(self.target_vocab)),
                name='training_decoder_inputs')
        # masking to prevent zero-padding from being used in loss/accuracy
        # scores
        training_decoder_mask = Masking(
                0.0,
                name='training_decoder_mask')(training_decoder_inputs)

        training_decoder_in = training_decoder_mask

        training_decoder_rnn = GRU(
                self.hps.dec_rnn_units,
                recurrent_dropout=self.hps.dec_rnn_dropout,
                return_sequences=True,
                return_state=True,
                name='training_decoder')
        training_decoder_outputs, _ = training_decoder_rnn(
            training_decoder_in,
            initial_state=training_initial_state)
        # activation will be softmax for one-hot data, sigmoid for k-hot
        training_decoder_dense = Dense(
                len(self.target_vocab),
                activation=self.activation,
                name='training_decoder_outputs')
        training_decoder_outputs = training_decoder_dense(
                training_decoder_outputs)

        self.training_model = Model(
            [training_encoder_inputs, training_decoder_inputs],
            training_decoder_outputs)

        #
        # inference models
        #

        # encode sequence input to state for decoder
        infer_encoder_model = Model(
                training_encoder_inputs,
                training_initial_state)

        # decode hidden state and incoming sequence to target plus hidden
        # state of decoder rnn
        infer_decoder_state_input = Input(shape=(self.hps.dec_rnn_units,))
        infer_decoder_outputs, infer_decoder_state = training_decoder_rnn(
            training_decoder_inputs, initial_state=infer_decoder_state_input
        )
        infer_decoder_outputs = training_decoder_dense(infer_decoder_outputs)
        infer_decoder_model = Model(
            [training_decoder_inputs, infer_decoder_state_input],
            [infer_decoder_outputs, infer_decoder_state]
        )

        self.infer_encoder_model = infer_encoder_model
        self.infer_decoder_model = infer_decoder_model

    def compile_models(self):
        loss = self.loss
        metrics = ['accuracy']
        if self.hps.optimizer == 'rmsprop':
            optimizer = optimizers.RMSProp
        elif self.hps.optimizer == 'adam':
            optimizer = optimizers.Adam
        optimizer_params = {
            k: getattr(self.hps, k) for k in ['lr', 'decay', 'clipvalue']
            if hasattr(self.hps, k)
        }
        self.training_model.compile(
            optimizer=optimizer(**optimizer_params),
            loss=loss,
            metrics=metrics)

    SERIALIZE_FIELDS = [
            "src_vocab", "target_vocab", "activation", "loss",
            "training_history", "hps"]

    def save(self, prefix):
        import pickle
        obj = {}
        for field in self.SERIALIZE_FIELDS:
            obj[field] = getattr(self, field)
        with open(prefix + "-obj.pickle", "wb") as fh:
            pickle.dump(obj, fh)
        self.save_model_weights(prefix)

    @classmethod
    def load(cls, prefix):
        import pickle
        with open(prefix + "-obj.pickle", "rb") as fh:
            obj_data = pickle.load(fh)
        obj = cls(**obj_data)
        obj.make_models()
        obj.load_model_weights(prefix)
        return obj

    @classmethod
    def load_from_package(cls, prefix):
        from pkg_resources import resource_stream as rs
        import pickle
        with rs(__name__, prefix + "-obj.pickle") as fh:
            obj_data = pickle.load(fh)
        obj = cls(**obj_data)
        obj.make_models()
        with rs(__name__, prefix + "-training.h5") as fh:
            obj.training_model.load_weights(fh)
        with rs(__name__, prefix + "-infer-encoder.h5") as fh:
            obj.infer_encoder_model.load_weights(fh)
        with rs(__name__, prefix + "-infer-decoder.h5") as fh:
            obj.infer_decoder_model.load_weights(fh)
        return obj

    def save_model_weights(self, prefix):
        self.training_model.save_weights(prefix + "-training.h5")
        self.infer_encoder_model.save_weights(prefix + "-infer-encoder.h5")
        self.infer_decoder_model.save_weights(prefix + "-infer-decoder.h5")

    def load_model_weights(self, prefix):
        self.training_model.load_weights(prefix + "-training.h5")
        self.infer_encoder_model.load_weights(prefix + "-infer-encoder.h5")
        self.infer_decoder_model.load_weights(prefix + "-infer-decoder.h5")

    def fit(self, *args, **kwargs):
        callbacks = []
        if 'callbacks' in kwargs:
            kwargs['callbacks'].extend(callbacks)
        else:
            kwargs['callbacks'] = callbacks
        history = self.training_model.fit(*args, **kwargs)
        self.training_history = history.history
        return history

    def infer(self, input_seq):
        "Infers initial state for decoder from sequence"
        return self.infer_encoder_model.predict(input_seq)

    def decode_sequence(self, state_value, target_seq, transform=lambda x: x,
                        maxlen=20):
        """Decodes a sequence given an initial state.

        Yields (1, 1, vocab_len) array with next predicted item from sequence.

        Parameters
        ----------
        state_value : numpy.array
            Initial state value for decoder
        target_seq : numpy.array
            Initial target sequence
        transform : function
            Function to transform sample from decoder before yielding and
            feeding back into the decoder. A good place to do your np.argmax
            or temperature or whatever.
        maxlen : int
            Maximum desired length of decoded sequence
        """

        yield transform(target_seq)
        for i in range(maxlen):
            output, h = self.infer_decoder_model.predict(
                [target_seq, state_value])
            sampled = transform(output)
            yield sampled
            target_seq = sampled
            state_value = h

    def vectorize_src(self, t, maxlen):
        """Vectorize list of tokens from source vocabulary.

        Returns (1, maxlen, src_vocab_len) array vectorizing t from source
        vocab.
        """

        seq = np.zeros((1, maxlen, len(self.src_vocab)))
        for i, item in enumerate(t):
            for j in item:
                seq[0, i, self.src_vocab_idx_map[j]] = 1.
        return seq

    def vectorize_target(self, t, maxlen):
        """Vectorize list of tokens from target vocabulary.

        Returns (1, maxlen, target_vocab_len) array vectorizing t from target
        vocab.
        """

        seq = np.zeros((1, maxlen, len(self.target_vocab)))
        for i, item in enumerate(t):
            for j in item:
                seq[0, i, self.target_vocab_idx_map[j]] = 1.
        return seq

    def to_src_vocab(self, arr, maxlen):
        """
        Converts (1, maxlen, src_vocab_len) array to list of lists of src_vocab
        items where the corresponding index is greater than zero.
        """
        out = []
        for timestep in arr[0]:
            out.append([self.src_vocab[i] for i in np.where(timestep > 0)[0]])
        return out

    def to_target_vocab(self, arr, maxlen):
        """
        Converts (1, maxlen, src_vocab_len) array to list of lists of src_vocab
        items where the corresponding index is greater than zero.
        """
        out = []
        for timestep in arr[0]:
            out.append(
                    [self.target_vocab[i] for i in np.where(timestep > 0)[0]])
        return out


def softmax_maximum(t):
    zeros = np.zeros(t.shape)
    argmax = np.argmax(t[0, -1, :])
    zeros[0, 0, argmax] = 1.
    return zeros


def softmax_temperature(t, temp=0.5):
    zeros = np.zeros(t.shape)
    dist = t[0, -1, :].astype(np.float64) + 0.0001
    pred = np.log(dist) / temp
    exp_pred = np.exp(pred)
    pred = exp_pred / (np.sum(exp_pred))
    yhat = np.argmax(np.random.multinomial(1, pred, 1))
    zeros[0, 0, yhat] = 1.
    return zeros


def sigmoid_top_n(t, n=4, minval=0.0, threshold=True):
    """Get top n values from array (for output of sigmoid layer)

    Returns t with all but the n highest values in the array zeroed out.
    Only values higher than minval (default 0) will be included (potentially
    resulting in fewer than n items being included in the result). If
    threshold is included, all nonzero values (after sorting and check
    against minval) will be converted to 1.

    """
    zeros = np.zeros(t.shape)
    top_n = np.argsort(t[0, -1, :])[::-1][:n]
    zeros[0, 0, top_n] = t[0, 0, top_n]
    if threshold:
        return np.where(zeros > minval, 1., 0.)
    else:
        return np.where(zeros > minval, t, 0.)


def sigmoid_sample(t, squish=1., threshold=True):
    """Sample from multi-label probabilities in t.

    The 'squish' parameter pushes the values in t toward extremes:
    Values > 1 emphasize high/low probabilities, while values < 1
    move toward uniform distribution."""

    # guard against divide by zero
    t = np.where(t == 0, 0.0001, t)
    t = np.where(t == 1, 0.9999, t)
    probs = 1 / (1 + (np.exp(np.log((1/t)-1)) + np.exp(squish)))
    if threshold:
        return np.where(t > np.random.uniform(size=probs.shape), 1, 0)
    else:
        return np.where(t > np.random.uniform(size=probs.shape), t, 0)
