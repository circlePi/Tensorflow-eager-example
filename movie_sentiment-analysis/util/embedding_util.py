import numpy as np
from tqdm import tqdm
import gensim
import config.text_cnn_config as config


def parse_word_vector(origin):
    if origin == 'glove':   # glove模型
        pre_trained_wordector = {}
        with open(config.glove_path, 'r') as fr:
            for line in fr:
                lines = line.strip('\n').split(' ')
                word = lines[0]
                vector = lines[1:]
                pre_trained_wordector[word] = vector
            return pre_trained_wordector
    elif origin == 'word2vector':   # google word2vector模型
        pre_trained_wordector = gensim.models.KeyedVectors.load_word2vec_format(config.word2vector_path, binary=True)
        return pre_trained_wordector
    else:
        raise ValueError('unknown word vector type!')


def get_embedding(model_type, word2id, embedding_dim=None):
    """
    :param mode_type: 嵌入层的类型
    :return: embedding_matrix
    嵌入层的作用：
    1. 将原始输入降维后作为神经网络的输入
    2. 提供语义信息
    嵌入层的类型主要有如下几种：
    CNN-rand：所有的word vector都是随机初始化的
    CNN-static：所有的word vector直接使用pre-trained词向量，并且不参与训练（trainable=False）
    CNN-non-static: 所有的word vector直接使用pre-trained词向量，但是参与训练，在训练过程会被fine-tuning
    CNN-multichannel：CNN-static和CNN-non-static的混合版本，即两种类型的输入
    see paper: Kim Y, Convolutional network for sentence classification, 2014
    论文得到的结论可做参考：
    CNN-staitc较CNN-rand好，这是因为static使用预训练的词向量包含语义信息
    CNN-non-static较CNN-static大部分要好，说明适当fine-tuning是有利的，可使vector更贴近具体任务
    CNN-multichannel较CNN-single在小规模数据集上有更好的表现，实际上这种方法是static和non-static方法的折中，
                    既不希望fine-tuning的值离pre-trained词向量太远，同时又保留一定的变化空间
    另一个值得注意的问题是词向量的选择，glove和word2vector，对于不同的任务效果不同
    """
    vocab_size = len(word2id)
    pre_trained_wordector = parse_word_vector(config.wordvector_origin)

    if model_type in ['CNN-static', 'CNN-non-Static']:
        if pre_trained_wordector is None:
            raise TypeError('error type')
        embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
        for word, id in tqdm(word2id.items()):
            try:
                word_vector = pre_trained_wordector[word]
                assert len(word_vector) == embedding_dim, \
                    'please check the consistency of embedding_dim and word vector shape'
                embedding_matrix[id-1] = word_vector  # 因为id从1开始的
            except KeyError:
                pass   # 若出现不在词向量里的单词，初始化为0向量（为零好还是随机好呢？）
        return embedding_matrix
    elif model_type == 'CNN-rand':
        embedding_matrix = np.random.randn(vocab_size, embedding_dim)
        return embedding_matrix
    elif model_type == 'CNN-multichannel':
        pass
    else:
        raise ValueError('unknown model type')










