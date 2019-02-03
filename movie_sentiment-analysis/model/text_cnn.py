"""
author: zelin
date: 2018-11-01
"""
import os
import time
import numpy as np
import tensorflow as tf
from util.embedding_util import get_embedding
from util.plot_util import loss_acc_plot
from util.lr_util import lr_update
import config.text_cnn_config as config


class TextCNN(tf.keras.Model):
    def __init__(self, num_classes,
                 checkpoint_dir,
                 vocab_size,
                 embedding_dim,
                 word2id,
                 model_type,
                 keep_dropout,
                 k_max_pooling):
        super().__init__()

        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word2id = word2id
        self.keep_dropout = keep_dropout
        self.k_max_pooling = k_max_pooling
        self.history = {}

        # Initial layer
        # embedding layer
        weights = get_embedding(model_type=model_type,
                                word2id=word2id,
                                embedding_dim=embedding_dim)
        if model_type == 'CNN-static':
            self.embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=weights, trainable=False)
        elif model_type == 'CNN-non-static':
            self.embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=weights, trainable=True)
        elif model_type == 'CNN-rand':
            self.embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=weights, trainable=True)
        elif model_type == 'CNN-multichannel':
            pass
        else:
            raise ValueError('unknown model type')
        # convolution layer
        self.conv0 = tf.layers.Conv2D(filters=config.filters,
                                      kernel_size=(config.kernel_size[0], self.embedding_dim),
                                      strides=config.strides,
                                      padding='valid',
                                      activation=None,
                                      kernel_initializer='he_normal')
        self.conv1 = tf.layers.Conv2D(filters=config.filters,
                                      kernel_size=(config.kernel_size[1], self.embedding_dim),
                                      strides=config.strides,
                                      padding='valid',
                                      activation=None,
                                      kernel_initializer='he_normal')
        self.conv2 = tf.layers.Conv2D(filters=config.filters,
                                      kernel_size=(config.kernel_size[2], self.embedding_dim),
                                      strides=config.strides,
                                      padding='valid',
                                      activation=None,
                                      kernel_initializer='he_normal')
        self.conv3 = tf.layers.Conv2D(filters=config.filters,
                                      kernel_size=(config.kernel_size[3], self.embedding_dim),
                                      strides=config.strides,
                                      padding='valid',
                                      activation=None,
                                      kernel_initializer='he_normal')
        self.Dense = tf.layers.Dense(self.num_classes)

    def pool(self, x):
        """
        pooling 层
        标配是1-max_pooling, 缺点是损失了很多结构信息
        改进的版本是k-max_pooling, 取top k 最大值
        """
        if self.k_max_pooling == 1:
            pool = tf.keras.layers.GlobalMaxPool2D()(x)
        else:
            pass
        return pool

    def call(self, inputs, training=None, mask=None):
        x = self.embeddings(inputs)         # (batch_size, sequence_length, embedding_dim)
        x = tf.expand_dims(x, -1)           # (batch_size, sequence_length, embedding_dim, 1)
        x0 = self.conv0(x)                  # (batch_size, sequence_length, 1, num_filters)
        x0 = tf.nn.relu(x0)
        x1 = self.conv1(x)
        x1 = tf.nn.relu(x1)
        x2 = self.conv2(x)
        x2 = tf.nn.relu(x2)
        x3 = self.conv3(x)
        x3 = tf.nn.relu(x3)

        max_pool_0 = self.pool(x0)         # (batch_size, num_filter)
        max_pool_1 = self.pool(x1)
        max_pool_2 = self.pool(x2)
        max_pool_3 = self.pool(x3)

        z = tf.concat([max_pool_0, max_pool_1, max_pool_2, max_pool_3], axis=1)
        z = tf.layers.flatten(z)
        z = tf.layers.dropout(z, rate=self.keep_dropout, training=training)
        output = self.Dense(z)
        return output

    def loss_fn(self, inputs, target, training):
        preds = self(inputs, training)
        # L2正则化
        loss_L2 = tf.add_n([tf.nn.l2_loss(v)
                            for v in self.trainable_variables
                            if 'bias' not in v.name]) * 0.001
        loss = tf.losses.sparse_softmax_cross_entropy(labels=target, logits=preds)
        loss = loss + loss_L2
        return loss

    def grads_fn(self, inputs, target, training):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(inputs, target, training)
        return tape.gradient(loss, self.variables)

    def save_model(self, model):
        """ Function to save trained model.
        """
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        checkpoint.save(file_prefix=checkpoint_prefix)

    def restore_model(self):
        # Run the model once to initialize variables
        dummy_input = tf.constant(tf.zeros((1, 1)))
        dummy_length = tf.constant(1, shape=(1,))
        self(dummy_input, dummy_length, False)
        # Restore the variables of the model
        saver = tf.contrib.Saver(self.variables)
        saver.restore(tf.train.latest_checkpoint
                      (self.checkpoint_directory))

    def get_accuracy(self, inputs, target, training):
        y = self(inputs, training)
        y_pred = tf.argmax(y, axis=1)
        correct = tf.where(tf.equal(y_pred, target)).numpy().shape[0]
        total = target.numpy().shape[0]
        return correct/total

    def fit(self, training_data, eval_data, pbar, num_epochs=100,
            early_stopping_rounds=5, verbose=1, train_from_scratch=True):
        """train the model"""
        if train_from_scratch is False:
            self.restore_model()

        # Initialize best loss. This variable will store the lowest loss on the
        # eval dataset.
        best_loss = 2018

        # Initialize classes to update the mean loss of train and eval
        train_loss = []
        eval_loss = []
        train_accuracy = []
        eval_accuracy = []

        # Initialize dictionary to store the loss history
        self.history['train_loss'] = []
        self.history['eval_loss'] = []
        self.history['train_accuracy'] = []
        self.history['eval_accuracy'] = []

        count = early_stopping_rounds

        # Begin training
        for i in range(num_epochs):
            # 在每个epoch训练之初初始化optimizer，决定是否使用学习率衰减
            learning_rate = lr_update(i+1, mode=config.lr_mode)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # Training with gradient descent
            start = time.time()
            for index, (sequence, label, _) in enumerate(training_data):
                # cpu需要类型转换，不然会报错：Could not find valid device
                sequence = tf.cast(sequence, dtype=tf.float32)
                label = tf.cast(label, dtype=tf.int64)
                grads = self.grads_fn(sequence, label, training=True)
                optimizer.apply_gradients(zip(grads, self.variables))
                pbar.show(index, use_time=time.time()-start)

            # Compute the loss on the training data after one epoch
            for sequence, label, _ in training_data:
                sequence = tf.cast(sequence, dtype=tf.float32)
                label = tf.cast(label, dtype=tf.int64)
                train_los = self.loss_fn(sequence, label, training=False)
                train_acc = self.get_accuracy(sequence, label, training=False)
                train_loss.append(train_los)
                train_accuracy.append(train_acc)
            self.history['train_loss'].append(np.mean(train_loss))
            self.history['train_accuracy'].append(np.mean(train_accuracy))

            # Compute the loss on the eval data after one epoch
            for sequence, label, _ in eval_data:
                sequence = tf.cast(sequence, dtype=tf.float32)
                label = tf.cast(label, dtype=tf.int64)
                eval_los = self.loss_fn(sequence, label, training=False)
                eval_acc = self.get_accuracy(sequence, label, training=False)
                eval_loss.append(eval_los)
                eval_accuracy.append(eval_acc)
            self.history['eval_loss'].append(np.mean(eval_loss))
            self.history['eval_accuracy'].append(np.mean(eval_accuracy))

            # Print train and eval losses
            if (i == 0) | ((i + 1) % verbose == 0):
                print('Epoch %d - train_loss: %4f - eval_loss: %4f - train_acc:%4f - eval_acc:%4f'
                      % (i + 1,
                         self.history['train_loss'][-1],
                         self.history['eval_loss'][-1],
                         self.history['train_accuracy'][-1],
                         self.history['eval_accuracy'][-1]))

            # Check for early stopping
            if self.history['eval_loss'][-1] < best_loss:
                best_loss = self.history['eval_loss'][-1]
                count = early_stopping_rounds
            else:
                count -= 1
            if count == 0:
                break
        # 画出loss_acc曲线
        loss_acc_plot(history=self.history)




