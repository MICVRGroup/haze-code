from dcnn import MyModel
from data_helper import load_data
import sys
sys.path.append('/share/software/anaconda3.0118/lib/python3.7/site-packages')
if __name__ == '__main__':
    test_size = 0.3
    image_height = 800
    image_width = 800
    image_in_channels = 3  # tong dao shu
    time_step = 3  # dui die shu
    batch_size = 21
    #conv_lstm_kernel = 3
    model_path = 'MODEL'
    learning_rate = 0.005
    epochs = 200


    data_loader = load_data(time_step=time_step)
    images_train, y_train, images_test, y_test, x_max, x_min, y_train_name, y_test_name \
        = data_loader.load_dataset(test_size=test_size)

    model = MyModel(
        image_height=image_height,
        image_width=image_width,
        image_in_channels=image_in_channels,
        time_step=time_step,
        batch_size=batch_size,
        #conv_lstm_kernel=conv_lstm_kernel,
        model_path=model_path,
        learning_rate=learning_rate,
        epochs=epochs)
    model.trian(images_train, y_train, images_test, y_test, x_max, x_min, y_train_name,
                y_test_name)
