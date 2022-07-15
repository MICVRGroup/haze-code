from data_helper import gen_batch
import tensorflow as tf
import tensorflow.contrib as contrib
import os
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import sys
from skimage import io
#sys.path.append('/share/software/anaconda3.0118/lib/python3.7/site-packages')

file_path = '/share/home/zhangyujie/zyj/wumai/daima/data2daima/0and255/lidu1/lidu15niant1b7zidong/'
if not os.path.exists(file_path):
    os.makedirs(file_path)

class MyModel():
    def __init__(self,
                 x_max=0.1,
                 x_min=0.1,
                 image_height=800,
                 image_width=800,
                 image_in_channels=3,
                 image_out_channels=1,
                 fusion_weight=None,
                 is_fusion_weight_manual=False,
                 time_step=4,
                 num_stack=3,
                 batch_size=2,
                 conv_lstm_kernel=3,
                 confusion_type='mean',  # mean sum
                 model_path='MODEL',
                 expand_method='manual',
                 layers_id=None,
                 learning_rate=0.005,
                 epochs=500, ):
        self.x_max=x_max
        self.x_min = x_min
        self.image_height = image_height
        self.image_width = image_width
        self.image_in_channels = image_in_channels
        self.image_height = image_height
        self.image_width = image_width
        self.image_out_channels = image_out_channels
        self.time_step = time_step
        self.fusion_weight = fusion_weight
        self.is_fusion_weight_manual = is_fusion_weight_manual
        self.num_stack = num_stack
        self.batch_size = batch_size
        self.conv_lstm_kernel = conv_lstm_kernel
        self.confusion_type = confusion_type
        self.model_path = model_path
        self.expand_method = expand_method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers_id = layers_id
        if not layers_id:
            self.layers_id = self.num_stack
        self._build_placeholder()
        self._build_net()

    def _build_placeholder(self):
        self.x_img = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.time_step, self.image_height, self.image_width, self.image_in_channels],
            name='x_img')
        self.x_img_tran = tf.transpose(self.x_img, [1, 0, 2, 3, 4])
        self.y = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.image_height, self.image_width, self.image_out_channels], name='y')
        if not self.is_fusion_weight_manual:  #  ru guo shi xun lian de dao quan zhong
            self.f_weight = tf.Variable(tf.truncated_normal(shape=[self.num_stack, 1, 1, 1, 1]))
        else:
            if len(self.fusion_weight) != self.num_stack:
                raise ValueError(f"quan zhong de ge shu bi xu wei {self.num_stack}ge, xian zai shi {len(self.fusion_weight)}ge, {self.fusion_weight}")
            self.f_weight_input = tf.placeholder(dtype=tf.float32,
                                                 shape=[self.num_stack], name='fusion_weight')

            self.f_weight = tf.reshape(self.f_weight_input, shape=[self.num_stack, 1, 1, 1, 1])
        if not self.is_fusion_weight_manual or np.abs(np.sum(self.fusion_weight)) - 1. > 0.001:
             self.f_weight = tf.nn.softmax(self.f_weight,0)


    def _build_net(self):
        def get_a_cell(name):
            cell_conv_lstm = contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                      input_shape=[self.image_height, self.image_width,
                                                                   self.image_in_channels],
                                                      output_channels=self.image_out_channels,
                                                      kernel_shape=[self.conv_lstm_kernel, self.conv_lstm_kernel],
                                                      name=name)
            return cell_conv_lstm

        with tf.name_scope("conlstm_net"):
            if self.expand_method == 'manual':
                print("================shou dong dui die ================")
                features = self.x_img_tran
                self.stacked_outputs = []
                for s in range(self.num_stack):
                    cell = get_a_cell(name='stack_{}'.format(s))
                    output, _ = tf.nn.dynamic_rnn(cell,
                                                  features,
                                                  dtype=tf.float32,
                                                  time_major=True)
                    self.stacked_outputs.append(output[-1])
                    features = output

                self.conv_lstm_outputs = tf.stack(self.stacked_outputs, axis=0) * self.f_weight

                if self.confusion_type == 'mean':
                    logits = tf.reduce_mean(self.conv_lstm_outputs, axis=0)

                elif self.confusion_type == 'sum':
                    logits = tf.reduce_sum(self.conv_lstm_outputs, axis=0)
                else:
                    logits = self.conv_lstm_outputs[-1]
                self.logits = logits
            else:
                print("================zi dong dui die ================")
                cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(name=str(i)) for i in range(self.num_stack)])
                output, final_state = tf.nn.dynamic_rnn(cell,
                                                        self.x_img_tran,
                                                        dtype=tf.float32,
                                                        time_major=True)
                self.logits = output[-1]
        with tf.name_scope("loss_net"):
            self.logits = self.logits  # *(self.x_max - self.x_min) + self.x_min
            self.loss = tf.reduce_mean(tf.square(self.logits - self.y))  # mse
        print(self.loss)

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
            test_loss_history = []
            for epoch in range(start_epoch, self.epochs):
                n_chunk = len(y_train) // self.batch_size
                if n_chunk * self.batch_size < len(y_train):
                    n_chunk += 1
                total_loss = 0
                batches = gen_batch([images_train, y_train_name], y_train, self.batch_size)
                for step, batch in enumerate(batches):

                    batch_x_img = batch[0][0]
                    batch_y_name = batch[0][1]
                    batch_y = batch[1]
                    if self.is_fusion_weight_manual:
                        feed_dict = {self.x_img: batch_x_img,
                                     self.y: batch_y,
                                     self.f_weight_input: self.fusion_weight}
                        print('-1')
                    else:
                        feed_dict = {self.x_img: batch_x_img,
                                     self.y: batch_y}

                    loss, _, f_weight = sess.run([self.loss, train_op,  self.f_weight],
                                                      feed_dict=feed_dict)

                    f_weight = f_weight.reshape(-1)
                    weights.append(f_weight)


                    total_loss += loss
                    if step % 2 == 0:
                        print(
                            "## Epoch:[{}/{}]--batch:[{}/{}]-- loss:{:.4}---{}".format(
                                epoch, self.epochs, step, n_chunk, loss,f_weight))
                        df = pd.DataFrame(data=weights)
                        print(x_max)
                        print(x_min)
                        df.to_excel(file_path+'lidu5shoudong811b21f_weight.xlsx')
                if epoch % 2 == 0:
                    print("### Saving model {}...".format(model_name + '-' + str(epoch)))
                    # saver.save(sess, os.path.join(self.model_path, model_name), global_step=epoch,
                    #            write_meta_graph=False)
                    loss_on_test = self.evaluate(images_test, y_test, sess, y_test_name, x_max, x_min)
                    loss_on_train = self.evaluate1(images_train, y_train, sess, y_train_name, x_max, x_min)

                    print("### Saving model")
                    test_loss_history.append(loss_on_test)

                    print("## test_loss_history:", test_loss_history)

                    self.evaluate1(images_train, y_train, sess, y_train_name, x_max, x_min)
                    self.save_pred_images1(images_train, y_train, sess, x_max, x_min, y_train_name,
                                          y_test_name)
                    self.evaluate(images_test, y_test, sess, y_test_name, x_max, x_min)
                    self.save_pred_images(images_test, y_test, sess, x_max, x_min, y_train_name,
                                          y_test_name)



    def evaluate(self, images_test, y_test, sess, y_test_name,x_max,x_min):
        total_square = 0
        total_loss = 0
        print(total_loss)
        batches = gen_batch([images_test, y_test_name], y_test, self.batch_size)
        y_pred = []
        for i, batch in enumerate(batches):
            batch_x_img = batch[0][0]
            batch_y_name = batch[0][1]
            batch_y = batch[1]
            if self.is_fusion_weight_manual:
                feed_dict = {self.x_img: batch_x_img,
                             self.y: batch_y,
                             self.f_weight_input: self.fusion_weight}
            else:
                feed_dict = {self.x_img: batch_x_img, self.y: batch_y}


            logits, loss = sess.run([self.logits, self.loss], feed_dict=feed_dict)
            self.save_stack_outputs(batch_x_img, batch_y_name, sess, batch_id=i)

            y_pred = logits * (x_max - x_min) + x_min
            y_true = batch_y * (x_max - x_min) + x_min
            print(x_max)
            print(x_min)
            total_square += np.sum((y_true - y_pred) ** 2)
            total_loss += loss
        mse = total_square / (len(y_test) * self.image_width * self.image_height  * self.image_out_channels)
        print(
            f"### On test data: Total loss {round(total_loss, 3)}, MSE {round(mse, 3)} RMSE {round(np.sqrt(mse), 3)} ")
        return round(total_loss, 3)
        #df.to_excel('./12score.xlsx')

    def save_stack_outputs(self, batch_x_img, batch_y_name, sess, batch_id):
        if self.expand_method != 'manual':
            print("zhi you shou dong dui die cai neng bao cun zhong jian ceng jie guo")
            return
        if self.is_fusion_weight_manual:
            feed_dict = {self.x_img: batch_x_img,
                         self.f_weight_input: self.fusion_weight}
        else:
            feed_dict = {self.x_img: batch_x_img}
        stacked_outputs = sess.run(self.stacked_outputs, feed_dict=feed_dict)
        for stack_id, layer_out in enumerate(stacked_outputs):
            for sample_id, step_out in enumerate(layer_out):
                np.save(
                    file_path+'test/feature/'+'layer_{}_time_{}'.format(
                    #'/share/home/zhangyujie/jiaquan/lidu5/zidongtiao/3ceng/result/b6/feature/output_batch_id_{}_layer_{}_time_{}'.format(
                   ## './output_batch_id_{}_layer_{}_time_{}'.format(
                        stack_id,
                        batch_y_name[sample_id][0]), step_out)
        print("bao cun convlstm zhong jian cen jie guo cheng gong")

    def save_pred_images(self, images_test, y_test, sess, x_max, x_min, y_train_name, y_test_name):
        from PIL import Image
        #file_path = '/share/home/zhangyujie/jiaquan/lidu5/zidongtiao/3ceng/result/b6/image'
        #file_path = '.'
        #if not os.path.exists(file_path):
            #os.makedirs(file_path)

        batches = gen_batch([images_test, y_test_name], y_test, self.batch_size)
        results = []
        for batch in batches:
            batch_x_img = batch[0][0]
            batch_y_name = batch[0][1]
            batch_y = batch[1]
            if self.is_fusion_weight_manual:
                feed_dict = {self.x_img: batch_x_img,
                             self.f_weight_input: self.fusion_weight}
            else:
                feed_dict = {self.x_img: batch_x_img}
            #y_pred = sess.run(self.prediction, feed_dict=feed_dict)
            logits = sess.run(self.logits, feed_dict=feed_dict)
            y_pred =logits * (x_max - x_min)+ x_min
            y_true =batch_y * (x_max - x_min) + x_min

            for i in range(len(batch_y)):
                img_pred, img_true, img_name = y_pred[i], y_true[i], batch_y_name[i][0]
                mse = np.sum((img_pred - img_true) ** 2) / (self.image_height * self.image_width * self.image_out_channels)
                rmse = np.sqrt(mse)
                results.append([img_name, mse, rmse])
                im_true = Image.fromarray(np.uint8(img_true.squeeze()))
                im_pred = Image.fromarray(np.uint8(img_pred.squeeze()))
                np.save(os.path.join(file_path, 'test/npy/', "true_{}".format(img_name)), np.uint8(img_true.squeeze()))
                np.save(os.path.join(file_path, 'test/npy/', "pred_{}".format(img_name)), np.uint8(img_pred.squeeze()))
                io.imsave(os.path.join(file_path, 'test/png/', "true_{}.png".format(img_name)),
                          np.uint8(img_true.squeeze()))
                io.imsave(os.path.join(file_path, 'test/png/', "pred_{}.png".format(img_name)),
                          np.uint8(img_pred.squeeze()))

                #im_true = Image.fromarray(np.uint8(img_true.squeeze()))
                #im_pred = Image.fromarray(np.uint8(img_pred.squeeze()))
                # np.save(os.path.join(file_path,'train/', "true_{}.npy".format(img_name)), np.uint8(img_true.squeeze()))
                # np.save(os.path.join(file_path,'train/', "pred_{}.npy".format(img_name)), np.uint8(img_pred.squeeze()))
                #im_true.save(os.path.join(file_path, 'train/', "true_{}".format(img_name)), quality=95)
                #im_pred.save(os.path.join(file_path, 'train/', "pred_{}".format(img_name)), quality=95)
            data = np.vstack(results)
            df = pd.DataFrame(data=data, columns=['name', 'mse', 'rmse'])
            df.to_excel(file_path + 'test/' + 'convlidu3t7b21wuqnian.xlsx')


    def evaluate1(self, images_train, y_train, sess, y_train_name,x_max,x_min):
        total_square = 0
        total_loss = 0
        print(total_loss)
        batches = gen_batch([images_train, y_train_name], y_train, self.batch_size)
        y_pred = []
        for i, batch in enumerate(batches):
            batch_x_img = batch[0][0]
            batch_y_name = batch[0][1]
            batch_y = batch[1]
            if self.is_fusion_weight_manual:
                feed_dict = {self.x_img: batch_x_img,
                             self.y: batch_y,
                             self.f_weight_input: self.fusion_weight}
            else:
                feed_dict = {self.x_img: batch_x_img, self.y: batch_y}


            logits, loss = sess.run([self.logits, self.loss], feed_dict=feed_dict)
            self.save_stack_outputs1(batch_x_img, batch_y_name, sess, batch_id=i)

            y_pred = logits * (x_max - x_min) + x_min
            y_true = batch_y * (x_max - x_min) + x_min
            print(x_max)
            print(x_min)
            total_square += np.sum((y_true - y_pred) ** 2)
            total_loss += loss
        mse = total_square / (len(y_train) * self.image_width * self.image_height  * self.image_out_channels)
        print(
            f"### On test data: Total loss {round(total_loss, 3)}, MSE {round(mse, 3)} RMSE {round(np.sqrt(mse), 3)} ")
        return round(total_loss, 3)
        #df.to_excel('./12score.xlsx')

    def save_stack_outputs1(self, batch_x_img, batch_y_name, sess, batch_id):
        if self.expand_method != 'manual':
            print("zhi you shou dong dui die cai neng bao cun zhong jian ceng jie guo")
            return
        if self.is_fusion_weight_manual:
            feed_dict = {self.x_img: batch_x_img,
                         self.f_weight_input: self.fusion_weight}
        else:
            feed_dict = {self.x_img: batch_x_img}
        stacked_outputs = sess.run(self.stacked_outputs, feed_dict=feed_dict)
        for stack_id, layer_out in enumerate(stacked_outputs):
            for sample_id, step_out in enumerate(layer_out):
                np.save(
                    file_path+'train/feature/'+'layer_{}_time_{}'.format(
                    #'/share/home/zhangyujie/jiaquan/lidu5/zidongtiao/3ceng/result/b6/feature/output_batch_id_{}_layer_{}_time_{}'.format(
                   ## './output_batch_id_{}_layer_{}_time_{}'.format(
                        stack_id,
                        batch_y_name[sample_id][0]), step_out)
        print("bao cun convlstm zhong jian cen jie guo cheng gong")

    def save_pred_images1(self, images_train, y_train, sess, x_max, x_min, y_train_name, y_test_name):
        from PIL import Image
        #file_path = '/share/home/zhangyujie/jiaquan/lidu5/zidongtiao/3ceng/result/b6/image'
        #file_path = '.'
        #if not os.path.exists(file_path):
            #os.makedirs(file_path)

        batches = gen_batch([images_train, y_train_name], y_train, self.batch_size)
        results = []
        for batch in batches:
            batch_x_img = batch[0][0]
            batch_y_name = batch[0][1]
            batch_y = batch[1]
            if self.is_fusion_weight_manual:
                feed_dict = {self.x_img: batch_x_img,
                             self.f_weight_input: self.fusion_weight}
            else:
                feed_dict = {self.x_img: batch_x_img}
            #y_pred = sess.run(self.prediction, feed_dict=feed_dict)
            logits = sess.run(self.logits, feed_dict=feed_dict)
            y_pred =logits * (x_max - x_min)+ x_min
            y_true =batch_y * (x_max - x_min) + x_min

            for i in range(len(batch_y)):
                img_pred, img_true, img_name = y_pred[i], y_true[i], batch_y_name[i][0]
                mse = np.sum((img_pred - img_true) ** 2) / (self.image_height * self.image_width * self.image_out_channels)
                rmse = np.sqrt(mse)
                results.append([img_name, mse, rmse])
                im_true = Image.fromarray(np.uint8(img_true.squeeze()))
                im_pred = Image.fromarray(np.uint8(img_pred.squeeze()))
                np.save(os.path.join(file_path, 'train/npy/', "true_{}".format(img_name)), np.uint8(img_true.squeeze()))
                np.save(os.path.join(file_path, 'train/npy/', "pred_{}".format(img_name)), np.uint8(img_pred.squeeze()))
                io.imsave(os.path.join(file_path, 'train/png/', "true_{}.png".format(img_name)),
                          np.uint8(img_true.squeeze()))
                io.imsave(os.path.join(file_path, 'train/png/', "pred_{}.png".format(img_name)),
                          np.uint8(img_pred.squeeze()))

                #im_true = Image.fromarray(np.uint8(img_true.squeeze()))
                #im_pred = Image.fromarray(np.uint8(img_pred.squeeze()))
                # np.save(os.path.join(file_path,'train/', "true_{}.npy".format(img_name)), np.uint8(img_true.squeeze()))
                # np.save(os.path.join(file_path,'train/', "pred_{}.npy".format(img_name)), np.uint8(img_pred.squeeze()))
                #im_true.save(os.path.join(file_path, 'train/', "true_{}".format(img_name)), quality=95)
                #im_pred.save(os.path.join(file_path, 'train/', "pred_{}".format(img_name)), quality=95)
            data = np.vstack(results)
            df = pd.DataFrame(data=data, columns=['name', 'mse', 'rmse'])
            df.to_excel(file_path + 'train/' + 'convlidu3t7b21wuqnian.xlsx')

if __name__ == '__main__':
    model = MyModel()

