import os
from tqdm import tqdm
import tensorflow as tf
from data_preprocessing.img_preprocess import zero_mean
from data_preprocessing.img_preprocess import random_brightness_image
from data_preprocessing.img_preprocess import random_crop_image


def img_tfrecord_encode(classes, tfrecord_filename, dir, file_names, is_training=True):
    """
    功能：读取图片转换成tfrecord格式的文件
    @params: classes: 标签类别
    @type：classes: dict
    @params: tfrecord_filename: tfrecord文件保存文件名
    @type：tfrecord_filename: str
    @params: path: 原始训练集存储路径
    """
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    for img_name in tqdm(file_names):
        name = img_name.split('.')[0]
        with tf.gfile.FastGFile(os.path.join(dir, img_name), 'rb') as gf:
            img = gf.read()
        if is_training:
            feature = {
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[classes[name]])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                'file_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name.encode()]))
            }
        else:
            feature = {
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[-1])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                'file_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name.encode()]))
            }
        example = tf.train.Example(features=tf.train.Features(feature=feature))  # example 对象将label和image数据进行封装
        writer.write(example.SerializeToString())   # 序列化为字符串
    writer.close()
    print('tfrecord writen done!')


def img_tfrecord_parse(tfrecord_filename, epochs, batch_size, shape,
                       padded_shapes=None, shuffle=True,
                       data_aug=False, buffer_size=1000):
    """
    @param: tfrecord_filename:tfrecord文件列表   @type:list
    @param: epoch:训练轮数（repeating次数）       @type:int
    @param：batch_size:批数据大小                @type:int
    @param: shape:图片维度                      @type:tuple
    @param: padded_shapes:不定长padding        @type:tuple
    @param: shuffle:是否打乱                   @type:boolean
    """

    def parse_example(serialized_example):
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                               'file_name': tf.FixedLenFeature([], tf.string)
                                           })
        # 解码
        image = tf.image.decode_jpeg(features['img_raw'])
        # 设置shape
        image = tf.image.resize_images(image, shape, method=1)
        if data_aug is True:
            image = random_brightness_image(image)
            image = random_crop_image(image, image_size=shape[0])
        label = tf.cast(features['label'], tf.int64)
        file_name = tf.cast(features['file_name'], tf.string)
        return image, label, file_name

    dataset = tf.data.TFRecordDataset(tfrecord_filename).map(parse_example)
    # dataset = dataset.map(lambda img, label, file_name: [zero_mean(img), label, file_name])

    if shuffle:
        if padded_shapes:
            dataset = dataset.repeat(epochs).shuffle(buffer_size=buffer_size).padded_batch(batch_size, padded_shapes)
        else:
            dataset = dataset.repeat(epochs).shuffle(buffer_size=buffer_size).batch(batch_size)
    else:
        if padded_shapes:
            dataset = dataset.repeat(epochs).padded_batch(batch_size, padded_shapes)
        else:
            dataset = dataset.repeat(epochs).batch(batch_size)
    # iterator = dataset.make_one_shot_iterator()
    # image, label, file_name = iterator.get_next()
    # return image, label, file_name
    return dataset


def text_tfrecord_encode(sentences, targets, classes, tfrecord_filename,
                   max_words_review=None):
    '''
       Function to parse each review in part and write to disk
       as a tfrecord.
       Args:
           sentences: [[],   [],  [],..]
           targets:   [tag1,tag2,tag3,...]
           tfrecord_filename: the paths to save tfrecord.
           vocabulary: list with all the words included in the vocabulary.
           word2idx: dictionary of words and their corresponding indexes
           max_words_review: max word number of each sequence.
       '''
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    for sent, tag in tqdm(zip(sentences, targets)):
        if max_words_review:
            sent = sent[:max_words_review]
        sequence_length = len(sent)
        frame_feature = list(map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), sent))
        feature = {
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[classes[tag]])),
            'sequence_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence_length]))}
        feature_lists = tf.train.FeatureLists(feature_list={
            'sequence': tf.train.FeatureList(feature=frame_feature)
        })
        example = tf.train.SequenceExample(context=tf.train.Features(feature=feature), feature_lists=feature_lists)
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()
    print('tfrecord writen done!')


def text_tfrecord_parse(tfrecord_filename, epochs, batch_size,
                        padding, shuffle=True, buffer_size=1000):

    def parse_example(serialized_example):
        context_features = {
            'label': tf.FixedLenFeature([], dtype=tf.int64),
            'sequence_length': tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            'sequence': tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features
        )
        label = context_parsed['label']
        sequence_length = context_parsed['sequence_length']
        sequence = sequence_parsed['sequence']
        return sequence, label, sequence_length

    dataset = tf.data.TFRecordDataset(tfrecord_filename).map(parse_example)
    if shuffle:
        if padding:
            dataset = dataset.repeat(epochs).shuffle(
                buffer_size=buffer_size).padded_batch(batch_size, padded_shapes=padding)
        else:
            dataset = dataset.repeat(epochs).shuffle(buffer_size=buffer_size).batch(batch_size)
    else:
        if padding:
            dataset = dataset.repeat(epochs).padded_batch(batch_size, padded_shapes=padding, padding_values=0)
        else:
            dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset







