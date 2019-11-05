Training your own model
=======================

You can train your own version of the models using the included
``pincelate.train`` module.

Run ``python -m pincelate.train --help`` for a list of options::

   --model-prefix MODEL_PREFIX
                           prefix for saved models (directories must already
                           exist!)
   --verbose             show keras progress bars (default to one line per
                           epoch)
   --random-state RANDOM_STATE
                           random state for train/test split
   --epochs EPOCHS       number of epochs to train
   --batch-size BATCH_SIZE
                           batch size
   --src {orth,phon}     source sequences
   --target {orth,phon}  target sequences
   --unidirectional      unidirectional rnn (default is bidirectional)
   --enc-rnn-units ENC_RNN_UNITS
                           units in encoder RNN
   --dec-rnn-units DEC_RNN_UNITS
                           units in decoder RNN
   --enc-rnn-dropout ENC_RNN_DROPOUT
                           recurrent dropout in encoder RNN
   --dec-rnn-dropout DEC_RNN_DROPOUT
                           recurrent dropout in decoder RNN
   --optimizer {adam,rmsprop}
                           optimizer (rmsprop or adam)
   --lr LR               learning rate for optimizer
   --decay DECAY         learning rate decay for optimizer
   --clipvalue CLIPVALUE
                           clip value for optimizer

A serialized model consists of a number of files, including the pickled hyperparameters and network weights. The ``--model-prefix`` option sets the path and first few characters for these files. For example, an option written like so::

   --model-prefix=my-models/phon2orth

... will direct the module to save files with names like
``my-models/phon2orth-obj.pickle``, ``my-models/phon2orth-training.h5``,
``my-models/phon2orth-infer-encoder.h5``, etc.

Pincelate needs both an orthography-to-phoneme model and a
phoneme-to-orthography model to operate; these are trained separately. You can
set the data for the encoder and decoder using the ``--src`` and ``--target``
options. For example, to train an orthography-to-phoneme model with 64 hidden
units in both the encoder and decoder::

   python -m pincelate.train --model-prefix=test-models/orth2phon --src=orth \
      --target=phon --enc-rnn-units=64 --dec-rnn-units=64

Training and test data from the CMU Pronouncing Dictionary is loaded and
prepared in ``pincelate.cmudictdata``.
