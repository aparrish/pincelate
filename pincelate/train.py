# specialized script for training cmudict data.


def main(args):
    from pincelate.seq2seq import Seq2Seq
    import pincelate.cmudictdata

    print("loading data...")
    data = pincelate.cmudictdata.load()
    tts = pincelate.cmudictdata.tts(data, random_state=args.random_state)

    if args.src == 'orth':
        src_vocab = data['orthography_vocab']
        encoder_data_train = tts['orth_data_train']
        encoder_data_valid = tts['orth_data_valid']
    elif args.src == 'phon':
        src_vocab = data['phoneme_feature_vocab']
        encoder_data_train = tts['phon_data_train']
        encoder_data_valid = tts['phon_data_valid']

    if args.target == 'orth':
        target_vocab = data['orthography_vocab']
        decoder_data_train = tts['orth_data_train']
        decoder_data_valid = tts['orth_data_valid']
        target_data_train = tts['orth_target_train']
        target_data_valid = tts['orth_target_valid']
        activation = 'softmax'
        loss = 'categorical_crossentropy'
    elif args.target == 'phon':
        target_vocab = data['phoneme_feature_vocab']
        decoder_data_train = tts['phon_data_train']
        decoder_data_valid = tts['phon_data_valid']
        target_data_train = tts['phon_target_train']
        target_data_valid = tts['phon_target_valid']
        activation = 'sigmoid'
        loss = 'binary_crossentropy'

    params = {}
    for param in ['enc_rnn_units', 'dec_rnn_units', 'enc_rnn_dropout',
                  'dec_rnn_dropout', 'lr', 'decay', 'clipvalue', 'optimizer']:
        params[param] = getattr(args, param)

    params['bidi'] = not(args.unidirectional)
    params['vae'] = False  # reserved for future use

    if args.verbose:
        print(params)

    model = Seq2Seq(
            src_vocab=src_vocab,
            target_vocab=target_vocab,
            activation=activation,
            loss=loss,
            hps=params)

    model.make_models()
    model.compile_models()
    model.training_model.summary()
    history = model.fit(
            [encoder_data_train, decoder_data_train],
            target_data_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(
                [encoder_data_valid, decoder_data_valid],
                target_data_valid),
            verbose=1 if args.verbose else 2)
    print(history.history)
    print("saving model...")
    model.save(args.model_prefix)
    print("done")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Training script for Pincelate')

    parser.add_argument(
            '--model-prefix',
            type=str,
            help='prefix for saved models (directories must already exist!)',
            required=True)
    parser.add_argument(
            '--verbose',
            action='store_true',
            help='show keras progress bars (default to one line per epoch)')
    parser.add_argument(
            '--random-state',
            type=int,
            default=None,
            help='random state for train/test split')
    parser.add_argument(
            '--epochs',
            type=int,
            default=50,
            help='number of epochs to train')
    parser.add_argument(
            '--batch-size',
            type=int,
            default=128,
            help='batch size')
    parser.add_argument(
            '--src',
            choices=['orth', 'phon'],
            required=True,
            help='source sequences')
    parser.add_argument(
            '--target',
            choices=['orth', 'phon'],
            required=True,
            help='target sequences')
    parser.add_argument(
            '--unidirectional',
            action='store_true',
            help='unidirectional rnn (default is bidirectional)')
    parser.add_argument(
            '--enc-rnn-units',
            type=int,
            default=128,
            help='units in encoder RNN')
    parser.add_argument(
            '--dec-rnn-units',
            type=int,
            default=128,
            help='units in decoder RNN')
    parser.add_argument(
            '--enc-rnn-dropout',
            type=float,
            default=0.2,
            help='recurrent dropout in encoder RNN')
    parser.add_argument(
            '--dec-rnn-dropout',
            type=float,
            default=0.2,
            help='recurrent dropout in decoder RNN')
    parser.add_argument(
            '--optimizer',
            type=str,
            choices=['adam', 'rmsprop'],
            default='adam',
            help='optimizer (rmsprop or adam)')
    parser.add_argument(
            '--lr',
            type=float,
            default=0.001,
            help='learning rate for optimizer')
    parser.add_argument(
            '--decay',
            type=float,
            default=1e-6,
            help='learning rate decay for optimizer')
    parser.add_argument(
            '--clipvalue',
            type=float,
            default=1.0,
            help='clip value for optimizer')

    args = parser.parse_args()
    main(args)
