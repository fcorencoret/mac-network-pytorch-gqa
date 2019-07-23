# File: utils.py
# Author: Ronil Pancholia
# Date: 4/22/19
# Time: 7:57 PM
import pickle
import sys

import numpy as np
import config

id2word = []
embedding_weights = None

def get_or_load_embeddings():
    global embedding_weights, id2word

    if embedding_weights is not None:
        return embedding_weights

    dataset_type = sys.argv[1]
    with open(f'data/{dataset_type}_dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    id2word = set(dic['word_dic'].keys())
    id2word.update(set(dic['answer_dic'].keys()))

    word2id = {word: id for id, word in enumerate(id2word)}

    embed_size = 300
    vocab_size = len(id2word)
    sd = 1 / np.sqrt(embed_size)
    embedding_weights = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
    embedding_weights = embedding_weights.astype(np.float32)

    with open("data/glove.6B.300d.txt", encoding="utf-8", mode="r") as textFile:
        for line in textFile:
            line = line.split()
            word = line[0]

            id = word2id.get(word, None)
            if id is not None:
                embedding_weights[id] = np.array(line[1:], dtype=np.float32)

    return embedding_weights

def params_to_dic():
    params = {
        "BASE_LR" : config.BASE_LR,
        "TRAIN_EPOCHS" : config.TRAIN_EPOCHS,
        "BATCH_SIZE" : config.BATCH_SIZE,
        "EARLY_STOPPING_ENABLED" : config.EARLY_STOPPING_ENABLED,
        "EARLY_STOPPING_PATIENCE" : config.EARLY_STOPPING_PATIENCE,

        ### Model Parameters
        "MAX_STEPS" : config.MAX_STEPS,
        "USE_SELF_ATTENTION" : config.USE_SELF_ATTENTION,
        "USE_MEMORY_GATE" : config.USE_MEMORY_GATE,
        "MAC_UNIT_DIM" : config.MAC_UNIT_DIM,

        ### Miscellaneous Config
        "MODEL_PREFIX" : config.MODEL_PREFIX,
        "RANDOM_SEED" : config.RANDOM_SEED,

        ## dropouts
        "encInputDropout" : config.encInputDropout,
        "encStateDropout" : config.encStateDropout,
        "stemDropout" : config.stemDropout,
        "qDropout" : config.qDropout,
        "qDropoutOut" : config.qDropoutOut,
        "memoryDropout" : config.memoryDropout,
        "readDropout" : config.readDropout,
        "writeDropout" : config.writeDropout,
        "outputDropout" : config.outputDropout,
        "controlPreDropout" : config.controlPreDropout,
        "controlPostDropout" : config.controlPostDropout,
        "wordEmbDropout" : config.wordEmbDropout,
    }
    return params

