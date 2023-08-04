import tensorflow as tf
import tensorflow.contrib as contrib
import os
from cnnconvdata_helper1 import gen_batch
import numpy as np
from skimage import io
import pandas as pd
import sys
sys.path.append('/share/software/anaconda3.0118/lib/python3.7/site-packages')

import openpyxl

file_path = '/share/home/zhangyujie/zyj/wumai/daima/data1daima/255dandunpy/dong/cnnconv/'
if not os.path.exists(file_path):
    os.makedirs(file_path)

class Mymodel:
    def __init__(self,
                 time_step=3,
                 num_factors=5,
                 img_width=920,
                 img_high=920,
                 img_channel=3,
                 img_channel_out=1,
                 expand_method='manual',
                 stack_layers_id=None,
                 conv_lstm_kernel=3,
                 num_stack_in_each_convlstm=2,
                 cnn_filters=32,
                 model_path='MODEL',
                 learning_rate=0.001,
                 epochs=10,
                 batch_size=2,

                 ):
        """
        x = np.array(x)  # (8, 3, 4, 920, 920, 3)  [yinsugeshu,batch_size,time_step,width,high,channels]
        y = np.array(y)  # (3, 920, 920, 3) [batch_size,width,high,channels]
        """
        self.time_step = time_step
        self.num_factors = num_factors
        self.img_width = img_width
        self.img_high = img_high
        self.img_channel = img_channel
        self.img_channel_out = img_channel_out
        self.conv_lstm_kernel = conv_lstm_kernel
        self.expand_method = expand_method
        self.stack_layers_id = stack_layers_id
        self.num_stack_in_each_convlstm = num_stack_in_each_convlstm
        self.cnn_filters = cnn_filters
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.build_placeholder()
        self.build_net()

    def build_placeholder(self):
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[self.num_factors, None, self.time_step, self.img_width, self.img_high,
                                       self.img_channel], name='input_x')
        self.y = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.img_width, self.img_high, self.img_channel_out], name='input_y')
 
    #def build_placeholder(self):
        #self.x_img_tran = tf.placeholder(dtype=tf.float32,
                                #shape=[None, self.time_step, self.img_width, self.img_high,self.img_channel], 
                                #name='input_x')
        #self.x = tf.transpose(self.x_img_tran, [1, 0, 2, 3, 4])  # time_major

        #self.y = tf.placeholder(dtype=tf.float32,
                                #shape=[None, self.img_width, self.img_high, self.img_channel_out], name='input_y')
   

    def build_net(self):
        for factor_idx in range(self.num_factors):  # bianlimeigeyinsu
            factor_inp = self.x[factor_idx]
            factor_inp_time_major = tf.transpose(factor_inp, [1, 0, 2, 3, 4])

        cell_conv_lstm = contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                  input_shape=[self.img_width, self.img_high, self.img_channel],
                                                  output_channels=self.img_channel_out,
                                                  kernel_shape=[self.conv_lstm_kernel, self.conv_lstm_kernel])
        output, final_state = tf.nn.dynamic_rnn(cell_conv_lstm,
                                                factor_inp_time_major,
                                                dtype=tf.float32,
                                                time_major=True)
        output = tf.layers.conv2d(output[-1], filters=32, kernel_size=[3, 3], name='output1',
                                   activation=tf.nn.relu,
                                   padding='SAME',
                                   kernel_initializer=tf.truncated_normal_initializer)

        self.logits = tf.layers.conv2d(output, filters=self.img_channel_out, kernel_size=[3, 3], name='output2',
                                  padding='SAME',
                                  kernel_initializer=tf.truncated_normal_initializer)  # juan jia he xing zhuang [2,2,64]  64 ge juan jia he chang kuan jun wei 2
        with tf.name_scope("loss_net"):
            self.loss = tf.reduce_mean(tf.square(self.logits - self.y))  # mse
        print(self.loss)
        #with tf.name_scope("loss_net"):
            #self.logits = tf.reshape(output, [-1, self.image_width * self.image_height, self.image_out_channels])
            #y_label = tf.reshape(self.y, [-1, self.image_width * self.image_height])
            #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label, logits=self.logits)
            #self.loss = tf.reduce_mean(cross_entropy)
        #with tf.name_scope('accuracy'):
            #self.prediction = tf.argmax(self.logits, axis=2, output_type=tf.int32)  # shape=(?, 640000),
            #self.correct_prediction = tf.equal(self.prediction, y_label)
            #self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train(self, images_train, y_train, y_train_name, images_test, y_test, y_test_name, x_max, x_min):
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
            test_loss_history = []
            for epoch in range(start_epoch, self.epochs):
                n_chunk = len(y_train) // self.batch_size
                if n_chunk * self.batch_size < len(y_train):
                    n_chunk += 1
                ave_loss = total_loss / n_chunk
                total_loss = 0

                batches = gen_batch(images_train, y_train, y_train_name, self.batch_size)
                for step, batch in enumerate(batches):
                    batch_x, batch_y, batch_name = batch[0], batch[1], batch[2]
                    feed_dict = {self.x: batch_x,
                                 self.y: batch_y}
                    loss, _ = sess.run([self.loss, train_op], feed_dict=feed_dict)
                    total_loss += loss
                    if step % 50 == 0:
                        print(
                            "## Epoch:[{}/{}]--batch:[{}/{}]--Last epoch loss ave:{:.4}--Current epoch loss:{:.4}".format(
                                epoch, self.epochs, step, n_chunk, ave_loss, loss))
                if epoch % 2 == 0:
                    print("### Saving model {}...".format(model_name + '-' + str(epoch)))
                    # saver.save(sess, os.path.join(self.model_path, model_name), global_step=epoch,
                    #            write_meta_graph=False)
                    loss_on_test = self.evaluate(images_test, y_test, sess, y_test_name, x_max, x_min)
                    test_loss_history.append(loss_on_test)
                    print("## test_loss_history:", test_loss_history)
                    self.save_pred_images(images_test, y_test, sess, y_test_name, x_max, x_min)

    def evaluate(self, images_test, y_test, sess, y_test_name, x_max, x_min):

        total_square = 0
        total_loss = 0
        batches = gen_batch(images_test, y_test, y_test_name, self.batch_size)
        for batch in batches:
            batch_x, batch_y, batch_name = batch[0], batch[1], batch[2]
            feed_dict = {self.x: batch_x, self.y: batch_y}
            logits, loss = sess.run([self.logits, self.loss], feed_dict=feed_dict)
            y_pred = logits * (x_max - x_min)+ x_min
            y_true = batch_y * (x_max - x_min) + x_min
            total_square += np.sum((y_true - y_pred) ** 2)
            total_loss += loss
        mse = total_square / (len(y_test) * self.img_width * self.img_high * self.img_channel_out)
        print(
            f"### On test data: Total loss {round(total_loss, 3)}, MSE {round(mse, 3)} RMSE {round(np.sqrt(mse), 3)} ")
        return round(total_loss, 3)

    def save_pred_images(self, images_test, y_test, sess, y_test_name, x_max, x_min):
        from PIL import Image
        batches = gen_batch(images_test, y_test, y_test_name, self.batch_size)
        results = []
        for batch in batches:
            batch_x, batch_y, batch_name = batch[0], batch[1], batch[2]
            feed_dict = {self.x: batch_x}
            logits = sess.run(self.logits, feed_dict=feed_dict)
            y_pred = logits * (x_max - x_min)+ x_min
            y_true = batch_y * (x_max - x_min) + x_min
            for i in range(len(batch_y)):
                img_pred, img_true, img_name = y_pred[i], y_true[i], batch_name[i][0]
                mse = np.sum((img_pred - img_true) ** 2) / (self.img_high * self.img_width * self.img_channel_out)
                rmse = np.sqrt(mse)
                results.append([img_name, mse, rmse])
                im_true = Image.fromarray(np.uint8(img_true.squeeze()))
                im_pred = Image.fromarray(np.uint8(img_pred.squeeze()))
                np.save(os.path.join(file_path, 'npy', "true_{}".format(img_name)),np.float32(img_true.squeeze()))
                np.save(os.path.join(file_path, 'npy', "pred_{}".format(img_name)),np.float32(img_pred.squeeze()))
                io.imsave(os.path.join(file_path, 'png', "true_{}.png".format(img_name[:-4])), np.uint8(img_true.squeeze()))
                io.imsave(os.path.join(file_path, 'png', "pred_{}.png".format(img_name[:-4])), np.uint8(img_pred.squeeze()))

        data = np.vstack(results)
        df = pd.DataFrame(data=data, columns=['name', 'mse', 'rmse'])
        df.to_excel(file_path + 'lidu75niancnn16t1b71ceng.xlsx')


if __name__ == '__main__':
    model = Mymodel()
