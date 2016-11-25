import re
import numpy as np

def clean_text(self ,raw_text):
    return [re.sub(r'([^\s\w]|_)+', '', str) for str in raw_text]

def char2index(text, char_dict):

        # convert to lower case
        lower = list(text.lower())

        # map to indices
        processed = map(lambda x: char_dict[x], lower)

        return processed


def load_glove(vocab):

    embedding_mat = [np.random.randn(50)]
    new_vocab= {}

    count = 1

    with open('glove.6B.50d.txt') as f:
        words_to_load = 400000

        for i, line in enumerate(f):

            if i >= words_to_load:
                break

            s = line.split()

            if s[0] in vocab:
                embedding_mat.append( np.asarray(s[1:]) )
                #new_vocab.append(s[0])
                new_vocab[s[0]] = count
                count += 1

    return new_vocab,np.array(embedding_mat)



def down_size_word2vec(vocab, w2v, new_w2v=None):

        lines = []

        new_voc = {}

        i = 0
        for word in vocab.keys():
            try:
                vec = w2v[word]
            except:
                continue

            string = ' '.join(map(str, vec))
            lines.append(word + ' ' + string + '\n')
            new_voc[word] = i
            i += 1


        return new_voc

def w2v_matrix(vocab_dict, w):

        w2v_mat = np.zeros((len(vocab_dict.keys()), 300))

        for k, v in vocab_dict.iteritems():
            w2v_mat[v] = w[k]

        return w2v_mat

def remove_zero_rows(X,y):
    mask = X.sum(axis=1) > 0

    return X[mask],y[mask.flatten()]