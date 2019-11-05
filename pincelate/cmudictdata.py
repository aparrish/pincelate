from sklearn.model_selection import train_test_split
import numpy as np
import pincelate.featurephone as featurephone
import pronouncing as pr
import re
import itertools

# pronouncing loads data on demand, so let's demand it
pr.init_cmu()


def phoneme_feature_vocab():
    feat_vals = set(itertools.chain(*featurephone.phone_feature_map.values()))
    all_vals = list(feat_vals) + ["str"]  # "str" used to mark stressed vowels
    feature_vocab = sorted(all_vals)
    feature_vocab_idx_map = {k: i for i, k in enumerate(feature_vocab)}
    return feature_vocab, feature_vocab_idx_map


def max_phoneme_feature_len():
    # +2 to account for beginning/ending tokens
    return max((len(p.split()) for w, p in pr.pronunciations)) + 2


def orthography_vocab():
    letters = set(itertools.chain(*[list(a) for a, b in pr.pronunciations]))
    # ensure vocab item 0 is end of string
    letter_vocab = ["$", "^"] + list(sorted(letters))
    letter_vocab_idx_map = {k: i for i, k in enumerate(letter_vocab)}
    return letter_vocab, letter_vocab_idx_map


def max_orthography_len():
    # +2 to account for beginning/ending tokens
    return max((len(w) for w, p in pr.pronunciations)) + 2


def load():
    pr.init_cmu()  # pronouncing loads data on demand, so let's demand it

    feature_vocab, feature_vocab_idx_map = phoneme_feature_vocab()
    letter_vocab, letter_vocab_idx_map = orthography_vocab()
    max_phone_len = max_phoneme_feature_len()
    max_letter_len = max_orthography_len()

    # TODO: held-out data here

    letter_data = np.zeros(
        (len(pr.pronunciations), max_letter_len, len(letter_vocab)),
        dtype=np.float32
    )
    phonfeat_data = np.zeros(
        (len(pr.pronunciations), max_phone_len, len(feature_vocab)),
        dtype=np.float32
    )
    letter_target_data = np.zeros(
        (len(pr.pronunciations), max_letter_len, len(letter_vocab)),
        dtype=np.float32
    )
    phonfeat_target_data = np.zeros(
        (len(pr.pronunciations), max_phone_len, len(feature_vocab)),
        dtype=np.float32
    )

    for i, (word, phones) in enumerate(pr.pronunciations):

        # orthography: one-hot for each character index
        word = "^" + word + "$"
        for t, char in enumerate(word):
            letter_data[i, t, letter_vocab_idx_map[char]] = 1.
            if t > 0:
                letter_target_data[i, t-1, letter_vocab_idx_map[char]] = 1.

        # clean errant comments
        phones = re.sub(' #.*$', '', phones)

        # phonemes: multi-label k-hot phonetic features
        for t, phone in enumerate(["^"] + phones.split() + ["$"]):
            for ft in featurephone.phone_feature_map[phone.strip('012')] + \
                    (('str',) if re.search(r'[12]$', phone) else tuple()):
                phonfeat_data[i, t, feature_vocab_idx_map[ft]] = 1.
                if t > 0:
                    phonfeat_target_data[i, t-1, feature_vocab_idx_map[ft]] = 1

    return {
        'phoneme_feature_vocab': feature_vocab,
        'phoneme_feature_idx_map': feature_vocab_idx_map,
        'orthography_vocab': letter_vocab,
        'orthography_idx_map': letter_vocab_idx_map,
        'phoneme_feature_data': phonfeat_data,
        'phoneme_feature_target_data': phonfeat_target_data,
        'orthography_data': letter_data,
        'orthography_target_data': letter_target_data
    }


def tts(data, test_size=0.2, random_state=None):
    tts = train_test_split(data['orthography_data'],
                           data['orthography_target_data'],
                           data['phoneme_feature_data'],
                           data['phoneme_feature_target_data'],
                           test_size=test_size,
                           random_state=random_state)
    return {
        "orth_data_train": tts[0],
        "orth_data_valid": tts[1],
        "orth_target_train": tts[2],
        "orth_target_valid": tts[3],
        "phon_data_train": tts[4],
        "phon_data_valid": tts[5],
        "phon_target_train": tts[6],
        "phon_target_valid": tts[7]
    }


if __name__ == '__main__':
    import sys
    import pickle
    obj = load()
    pickle.dump(obj, sys.stdout.buffer)
