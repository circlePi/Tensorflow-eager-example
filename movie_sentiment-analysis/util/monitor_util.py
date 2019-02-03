"""进度条监控batch进度"""
import os
import tensorflow as tf


class ProgressBar(object):
    def __init__(self, data_size, batch_size, width=50):
        self.data_size = data_size
        self.batch_size = batch_size
        self.width = width

    def show(self, batch_index, use_time):
        """控制台显示 batch 训练进度"""
        num_batch = self.data_size / self.batch_size
        num = (batch_index + 1) * self.batch_size
        if num > self.data_size:
            num = self.data_size
        char_num = int(batch_index * self.width / num_batch)
        ratio = int(100 * num / self.data_size)
        show_bar = ('[%%-%ds]' % self.width) % (char_num * ">")
        show_str = 'Batch %d%% %s %d/%d (%.1fs used)'
        print(show_str % (ratio, show_bar,
                          num, self.data_size, use_time), end='\r')


class WriteSummary(object):
    pass




