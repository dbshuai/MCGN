# encoding:utf-8
import numpy as np
import os
import utils

np.random.seed(123456)
domain = "res"


#################
# read embeddings
#################
def read_emb_idx(filename, word_to_idx, dim):
    with open(filename, 'r') as f:
        embeddings = np.zeros((len(word_to_idx), dim))
        for line in f:
            line = line.strip()
            one = line.split(' ')
            word = one[0]
            emb = [float(i) for i in one[1:]]
            if len(emb) == dim and word in word_to_idx:
                embeddings[word_to_idx[word]] = np.array(emb)

        ''' Add padding and unknown word to embeddings and word2idx'''
        embeddings[0] = np.zeros(dim)  # _padding
        embeddings[1] = np.random.random(dim)  # _unk

        print("Finish loading embedding %s * * * * * * * * * * * *" % filename)
        return embeddings


###########
# read datasets
###########
def read_data1(path):
    """
    读取包含sentence、opinion、target、target_polartiy的文件
    sentence:word seg
    opinion:0 1 2: O B I
    target:0 1 2: O B I
    :param path:
    :return:dataset {
        "opinion":[[0,0,1,2],]
        "sentence":[[w1,w2,w3,w4],]
        "target":[[1,2,0,0],]
        "target_polarity":[1,1,0,0]
        "target_opinion":[[1,2,3,4,0],]
        "opinion_ex":[[[1,0,0],],]
        "aspect_ex":[[[1,0,0],],]
    }
    """
    dataset = {}
    names = ["opinion", "sentence", "target", "target_polarity"]
    for name in names:
        data = []
        file_name = path + name + ".txt"
        with open(file_name, "r") as f:
            for line in f:
                line = line.strip().split(" ")
                if name != "sentence":
                    line = [int(x) for x in line]
                data.append(line)
            dataset[name] = data
    dataset["target_opinion"] = []
    dataset["opinion_ex"] = []
    dataset["aspect_ex"] = []
    for target, opinion in zip(dataset["target"], dataset["opinion"]):
        target_opinion = target.copy()
        opinion_ex = []
        aspect_ex = []
        for i in range(len(opinion)):
            if opinion[i] == 1:
                target_opinion[i] = 3
                opinion_ex.append([0, 1, 0])
            elif opinion[i] == 2:
                target_opinion[i] = 4
                opinion_ex.append([0, 0, 1])
            else:
                opinion_ex.append([1, 0, 0])
        for i in range(len(opinion)):
            if target[i] == 1:
                aspect_ex.append([0,1,0])
            elif target[i] == 2:
                aspect_ex.append([0,0,1])
            else:
                aspect_ex.append([1,0,0])
        dataset["target_opinion"].append(target_opinion)
        dataset["opinion_ex"].append(opinion_ex)
        dataset["aspect_ex"].append(aspect_ex)
    return dataset


def read_data2(path):
    """
    :param path:
    :return: dataset {
        "sentence":[[w1,w2,w3,w4],]
        "doamin":[0,1,]0:yelp 1:electronics
        "label":[1.0,2.0,] 1.0~5.0
    }
    """
    dataset = {"sentence": [], "domain": [], "label": []}
    domains = ["yelp_large/", "electronics_large/"]
    names = ["label", "text"]
    for domain in domains:
        for name in names:
            file_name = path + domain + name + ".txt"
            with open(file_name, "r") as f:
                for line in f:
                    if name == "text":
                        line = line.strip().split(" ")
                        dataset["sentence"].append(line)
                        if domain == "yelp_large/":
                            dataset["domain"].append(0)
                        else:
                            dataset["domain"].append(1)
                    else:
                        line = float(line.strip())
                        if line>3:
                            dataset["label"].append(0)
                        elif line == 3:
                            dataset["label"].append(2)
                        else:
                            dataset["label"].append(1)
    return dataset


def get_vocab(datasets):
    vocab = []
    for dataset in datasets:
        sentences = dataset["sentence"]
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab.append(word)
    return vocab


def get_word_to_idx(vocab):
    word_to_idx = {}
    for i, word in enumerate(vocab):
        word_to_idx[word] = i + 2
        word_to_idx["_padding"] = 0  # PyTorch Embedding lookup need padding to be zero
        word_to_idx["_unk"] = 1
    idx_to_word = dict((word_to_idx[word], word) for word in word_to_idx)
    return word_to_idx, idx_to_word


def sentence_to_index(word2idx, sentences):
    """
    Transform sentence into lists of word index
    :param word2idx:
        word2idx = {word:idx, ...}
    :param sentences:
        list of sentences which are list of word
    :return:
    """
    print("-------------begin making sentence xIndexes-------------")
    sentences_indexes = []
    unk = 0
    for sentence in sentences:
        s_index = []
        for word in sentence:
            word = word
            if word == "\n":
                continue
            if word in word2idx:
                s_index.append(word2idx[word])
            else:
                s_index.append(word2idx["_unk"])
                unk += 1
                print("  --", word, "--  ")

        if len(s_index) == 0:
            print(len(sentence), "+++++++++++++++++++++++++++++++++empty sentence")
            s_index.append(word2idx["_unk"])
        sentences_indexes.append(s_index)
    assert len(sentences_indexes) == len(sentences)
    print("-------------finish making sentence xIndexes-------------", "unk=", unk)
    return sentences_indexes


def make_datasets(datasets, word2idx):
    # for i in ["train_sent", "train_doc", "test_sent"]:
    for i in ["train_sent", "test_sent"]:
        sentences = datasets[i]["sentence"]
        xIndexes = sentence_to_index(word2idx, sentences)
        datasets[i]["index"] = xIndexes
    return datasets


def processing():
    dir_sent_train = f"../data/data_preprocessed/{domain}/train/"
    dir_sent_test = f"../data/data_preprocessed/{domain}/test/"
    # dir_doc = "../data/data_doc/"
    dir_output = "../data/pkl/IMN/"
    if not os.path.exists(dir_output):
        os.mkdir(dir_output)
    print("reading data file...")
    dataset_sent_train = read_data1(dir_sent_train)
    dataset_sent_test = read_data1(dir_sent_test)
    # dataset_doc = read_data2(dir_doc)
    # datasets = [dataset_sent_train, dataset_sent_test, dataset_doc]
    datasets = [dataset_sent_train, dataset_sent_test]
    print("making vocab...")
    vocab = get_vocab(datasets)
    word2idx, idx2word = get_word_to_idx(vocab)
    # read the embedding files
    print("making embedding")
    emb_dir = "../data/embeddings/"
    general_emb_file = emb_dir + "glove.840B.300d.txt"
    domain_emb_file = emb_dir + f"{domain}.txt"
    general_embeddings = read_emb_idx(general_emb_file, word2idx, 300)
    domain_embeddings = read_emb_idx(domain_emb_file, word2idx, 100)
    # dataset_total = {"train_sent": dataset_sent_train, "train_doc": dataset_doc, "test_sent": dataset_sent_test}
    dataset_total = {"train_sent": dataset_sent_train, "test_sent": dataset_sent_test}
    # transform sentence to word index
    datasets = make_datasets(dataset_total, word2idx)
    # dir_output the transformed files
    print("making pickle file...")
    utils.pkl_dump(datasets, dir_output + "/features.pkl")
    utils.pkl_dump(word2idx, dir_output + "/word2idx.pkl")
    utils.pkl_dump(general_embeddings, dir_output + "/general_embeddings.pkl")
    utils.pkl_dump(domain_embeddings, dir_output + "/domain_embeddings.pkl")
    utils.pkl_dump(idx2word, dir_output + "/idx2word.pkl")


if __name__ == "__main__":
    processing()
