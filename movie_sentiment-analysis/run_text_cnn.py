from config import text_cnn_config as config
from util.data_util import get_text_tfrecord, text_get_dataset
from util.monitor_util import ProgressBar
from model.text_cnn import TextCNN
import tensorflow as tf

# 动态图
tf.enable_eager_execution()
# 文本转换成TFrecord格式
vocab, word2id, train_size = get_text_tfrecord(tfrecord_filename=config.tfrecord_filename, classes=config.classes)
# 初始化进度条
pbar = ProgressBar(train_size, config.batch_size)
# 获取训练集和验证集
train_dataset, valid_dataset = text_get_dataset(tfrecord_filename=config.tfrecord_filename,
                                                epochs=1,
                                                batch_size=config.batch_size)
# 初始化模型
model = TextCNN(num_classes=len(config.classes),
                checkpoint_dir=config.checkpoint_dir,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                word2id=word2id,
                model_type=config.model_type,
                keep_dropout=config.keep_dropout,
                k_max_pooling=config.k_max_pooling)
# 训练
model.fit(training_data=train_dataset,
          eval_data=valid_dataset,
          pbar=pbar,
          num_epochs=config.epochs,
          early_stopping_rounds=10,
          verbose=1)
# 模型保存
model.save_model(model=model)
