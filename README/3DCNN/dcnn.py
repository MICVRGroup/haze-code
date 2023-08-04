from data_helper import gen_batch
import tensorflow as tf
import tensorflow.contrib as contrib
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import sys
sys.path.append('/share/software/anaconda3.0118/lib/python3.7/site-packages')

class MyModel():
    def __init__(self,
                 image_height=800,
                 image_width=800,
                 image_in_channels=3,
                 image_out_channels=1,
                 num_stack=3,
                 time_step=4,
                 filter_depth=3,
                 batch_size=2,
                 kernel_size=3,
                 num_filters=64,
                 model_path='MODEL',
                 learning_rate=0.005,
                 epochs=500, ):
        self.image_height = image_height
        self.image_width = image_width
        self.image_in_channels = image_in_channels
        self.image_height = image_height
        self.image_width = image_width
        self.image_out_channels = image_out_channels
        self.num_stack = num_stack
        self.time_step = time_step
        self.filter_depth = filter_depth
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._build_placeholder()
        self._build_net()

    def _build_placeholder(self):
        self.x_img = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.time_step, self.image_height, self.image_width, self.image_in_channels],
            name='x_img')
        #self.x_img_tran = tf.transpose(self.x_img, [1, 0, 2, 3, 4])
        self.y = tf.placeholder(
            dtype=tf.int32,
            shape=[None, self.image_height, self.image_width, self.image_out_channels], name='y')



    def _build_net(self):
        weight = tf.truncated_normal(
            shape=[self.filter_depth, self.kernel_size, self.kernel_size, self.image_in_channels, self.num_filters])
        outputs = tf.nn.conv3d(self.x_img, weight, strides=[1, 1, 1, 1, 1], padding='SAME')
        outputs = tf.transpose(outputs, [0, 2, 3, 1, 4])  # [?,800,800,4,64]
        outputs = tf.reshape(outputs, [-1, self.image_height, self.image_width, self.num_filters * self.time_step])
        outputs = tf.layers.conv2d(outputs, filters=2, kernel_size=[3, 3], name='output1',
                                   activation=tf.nn.relu,
                                   padding='SAME',
                                   kernel_initializer=tf.truncated_normal_initializer)
        print(outputs.shape)
        self.logits = tf.layers.conv2d(outputs, filters=2, kernel_size=[3, 3], name='output2',
                                   padding='SAME',
                                   kernel_initializer=tf.truncated_normal_initializer)
        print(outputs.shape)
        self.stacked_outputs =self.logits
            #self.x_img_tran = output
            #self.stacked_outputs.append(output[-1])
        with tf.name_scope("loss_net"):
            flatten_logits = tf.reshape(self.logits, [-1, self.image_width * self.image_height, 2])
            print(flatten_logits.shape)
            y_label = tf.reshape(self.y, [-1, self.image_width * self.image_height])
            print(y_label.shape)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label, logits=flatten_logits)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('accuracy'):
            self.prediction = tf.argmax(flatten_logits, axis=2, output_type=tf.int32)  # shape=(?, 640000),
            self.correct_prediction = tf.equal(self.prediction, y_label)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    #
    def trian(self, images_train, y_train, images_test, y_test, x_max, x_min, y_train_name,
              y_test_name):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.apply_gradients(zip(gradients, params))
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
        model_name = 'step{}'.format(self.time_step)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start_epoch = 0
            check_point = tf.train.latest_checkpoint(self.model_path)
            if check_point:
                saver.restore(sess, check_point)
                start_epoch += int(check_point.split('-')[-1])
                if start_epoch % 2 != 0:
                    start_epoch += 1
                print("### Loading exist model <{}> successfully...".format(check_point))
            total_loss = 0
            weights = []
            for epoch in range(start_epoch, self.epochs):
                n_chunk = len(y_train) // self.batch_size
                if n_chunk * self.batch_size < len(y_train):
                    n_chunk += 1
                total_loss = 0
                batches = gen_batch([images_train, y_train_name], y_train, self.batch_size)
                for step, batch in enumerate(batches):

                    batch_x_img = batch[0][0]
                    batch_y_name = batch[0][1]
                    batch_y = batch[1].reshape(-1,
                                               self.image_height,
                                               self.image_width,
                                               self.image_out_channels)

                    feed_dict = {self.x_img: batch_x_img,
                                 self.y: batch_y}

                    loss, _, acc = sess.run([self.loss, train_op, self.accuracy],
                                                      feed_dict=feed_dict)


                    total_loss += loss
                    if step % 2 == 0:
                        print(
                            "## Epoch:[{}/{}]--batch:[{}/{}]-- loss:{:.4}---Acc:{:.4}".format(
                                epoch, self.epochs, step, n_chunk, loss, acc))
                if epoch % 2 == 0:
                    print("### Saving model {}...".format(model_name + '-' + str(epoch)))
                    # saver.save(sess, os.path.join(self.model_path, model_name), global_step=epoch,
                    #            write_meta_graph=False)
                    self.evaluate(images_test, y_test, sess, y_test_name)
                    self.save_pred_images(images_test, y_test, sess, x_max, x_min, y_train_name,
                                          y_test_name)


    def evaluate(self, images_test, y_test, sess, y_test_name):

        batches = gen_batch([images_test, y_test_name], y_test, self.batch_size)
        y_pred = []
        for i, batch in enumerate(batches):
            batch_x_img = batch[0][0]
            batch_y_name = batch[0][1]
            batch_y = batch[1].reshape(-1, self.image_height * self.image_width)
            feed_dict = {self.x_img: batch_x_img}

            prediction = sess.run(self.prediction, feed_dict=feed_dict)
            self.save_stack_outputs(batch_x_img, batch_y_name, sess, batch_id=i)

            y_pred.append(prediction)
        y_pred = np.vstack(y_pred)  # [?,640000)
        y_true = y_test.reshape(-1, self.image_height * self.image_width)
        scores = []
        for i in range(len(y_pred)):
            scores.append([accuracy_score(y_pred[i], y_true[i]), recall_score(y_pred[i], y_true[i]),
                           precision_score(y_pred[i], y_true[i]), f1_score(y_pred[i], y_true[i])])
        data = np.vstack(scores)
        names = np.vstack(y_test_name)
        data = np.hstack([names, data])
        df = pd.DataFrame(data=data, columns=['name', 'acc', 'recall', 'precision','fscore'])
        df.to_excel('/share/home/zhangyujie/zyj/yuntu/dcnn/result/b21t3dcnnscore.xlsx')
        #df.to_excel('./12score.xlsx')

    def save_stack_outputs(self, batch_x_img, batch_y_name, sess, batch_id):
        feed_dict = {self.x_img: batch_x_img}
        stacked_outputs = sess.run(self.stacked_outputs, feed_dict=feed_dict)
        for sample_id, step_out in enumerate(stacked_outputs):
            np.save(
                '/share/home/zhangyujie/zyj/yuntu/dcnn/result/feature/time_{}'.format(
                   # './output_batch_id_{}_layer_{}_time_{}'.format(
                batch_y_name[sample_id][0]), step_out)
        print("bao cun convlstm zhong jian cen jie guo cheng gong")

    def save_pred_images(self, images_test, y_test, sess, x_max, x_min, y_train_name, y_test_name):
        from PIL import Image
        file_path = '/share/home/zhangyujie/zyj/yuntu/dcnn/result/png'
        #file_path = '.'
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        batches = gen_batch([images_test, y_test_name], y_test, self.batch_size)
        for batch in batches:
            batch_x_img = batch[0][0]
            batch_y_name = batch[0][1]
            batch_y = batch[1]
            feed_dict = {self.x_img: batch_x_img}
            y_pred = sess.run(self.prediction, feed_dict=feed_dict)
            for i in range(len(batch_y)):
                a = batch_y[i]
                im_true = np.uint8(a)
                im_true = Image.fromarray(im_true)
                plt.imsave(os.path.join(file_path, "true_{}.jpeg".format(batch_y_name[i][0])), im_true, cmap='bone')
                im_pred = (y_pred[i] * (x_max - x_min) + x_min).reshape(self.image_height, self.image_width)
                im_pred = np.uint8(im_pred)
                im_pred = Image.fromarray(im_pred)
                plt.imsave(os.path.join(file_path, "pred_{}.jpeg".format(batch_y_name[i][0])), im_pred, cmap='bone')


if __name__ == '__main__':
    model = MyModel()

