import tensorflow as tf
import config.text_cnn_config as config


def lr_update(epoch, mode=None):
    """训练过程中学习率的更新"""
    # 指数衰减
    if mode == 'exp_decay':
        learning_rate = tf.train.exponential_decay(config.learning_rate,
                                                   epoch, 100, 0.96,
                                                   staircase=True)
    # 周期性学习率
    elif mode == 'clr':
        print('=============CLR模式===============')
        high = float(input('请指定学习率上限:'))
        low = float(input('请指定学习率下限:'))
        pass
    else:
        learning_rate = config.learning_rate
    return learning_rate