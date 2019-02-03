import os
from sklearn.model_selection import train_test_split
from util.tfrecord_util import img_tfrecord_encode, img_tfrecord_parse
from util.tfrecord_util import text_tfrecord_encode, text_tfrecord_parse
from data_preprocessing.text_preprocess import bulid_vocab


def train_val_split(train_dir, test_size, train_use_size=None, random_state=2018):
    if train_use_size:
        train_path = os.listdir(train_dir)[:train_use_size]
    else:
        train_path = os.listdir(train_dir)
    targets = [path.split('.')[0] for path in train_path]
    x_train, x_valid, y_train, y_valid = train_test_split(
        train_path, targets, test_size=test_size, random_state=random_state, shuffle=True)
    return x_train, x_valid     # type: list


def get_img_tfrecord(train_dir, classes, tfrecord_filename):
    x_train, x_valid = train_val_split(train_dir, test_size=0.3, train_use_size=100)
    train_size = len(x_train)
    valid_size = len(x_valid)
    img_tfrecord_encode(classes, tfrecord_filename['train'], train_dir, x_train)
    print('train--image to tfrecord done, shape is %d!' % (len(x_train)))
    img_tfrecord_encode(classes, tfrecord_filename['valid'], train_dir, x_valid)
    print('valid--image to tfrecord done, shape is %d!' % (len(x_valid)))
    return train_size, valid_size


def get_img_dataset(tfrecord_filename, epochs, batch_size, shape,):
    train_dataset = img_tfrecord_parse(tfrecord_filename['train'], epochs, batch_size, shape, data_aug=True)
    valid_dataset = img_tfrecord_parse(tfrecord_filename['valid'], epochs, batch_size, shape)
    return train_dataset, valid_dataset


def get_text_tfrecord(tfrecord_filename, classes):
    vocab, word2id, x_train, \
    x_valid, y_train, y_valid = bulid_vocab(
                                    min_freq=1,
                                    stop_list=None,
                                    test_size=0.3,
                                    train_use_size=None)

    text_tfrecord_encode(x_train,
                         y_train,
                         classes,
                         tfrecord_filename['train'],
                         max_words_review=None)
    print('train--text to tfrecord done, shape is %d!' % (len(x_train)))

    text_tfrecord_encode(x_valid,
                         y_valid,
                         classes,
                         tfrecord_filename['valid'],
                         max_words_review=None)
    print('valid--text to tfrecord done, shape is %d!' % (len(x_valid)))
    return vocab, word2id, len(x_train)


def text_get_dataset(tfrecord_filename, epochs, batch_size):
    # padding 中的None指的是以当前batch的最大维度填充，numeric默认填充0，string默认填充空字符串
    # textCnn可以处理不定长文本，这里可以不padding
    # 当然padding也无伤大雅，因为textCNN的好基友是max_pooling，所以padding成0（空字符串对应0向量）不影响
    train_dataset = text_tfrecord_parse(tfrecord_filename['train'], epochs, batch_size, padding=([None], [], []))
    valid_dataset = text_tfrecord_parse(tfrecord_filename['valid'], epochs, batch_size, padding=([None], [], []))
    return train_dataset, valid_dataset






