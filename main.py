from model import MyModel
from data_helper import load_data
import sys
#sys.path.append('/share/software/anaconda3.0118/lib/python3.7/site-packages')

if __name__ == '__main__':
    test_size = 0.3
    image_height = 421
    image_width = 234
    image_in_channels = 5  # tong dao shu
    time_step = 1
    num_stack = 3  # dui die shu
    batch_size = 7
    layers_id = [1,2,3]
    confusion_type = 'mean'  # mean and sum   other san zhong
    conv_lstm_kernel = 3
    model_path = 'MODEL'
    learning_rate = 0.005
    epochs = 300
    expand_method = 'manual'  # 'manual' or 'auto'

    data_loader = load_data(time_step=time_step)
    images_train, y_train, images_test, y_test, x_max, x_min, y_train_name, y_test_name \
        = data_loader.load_dataset(test_size=test_size)

    model = MyModel(
        x_max=x_max,
        x_min=x_min,
        image_height=image_height,
        image_width=image_width,
        image_in_channels=image_in_channels,
        fusion_weight=[0.1, 0.1, 0.8],    # quan zhong de ge shu bi xu deng yu num_stack
        is_fusion_weight_manual=False,    #True or False
        time_step=time_step,
        num_stack=num_stack,
        layers_id=layers_id,
        batch_size=batch_size,
        expand_method=expand_method,
        conv_lstm_kernel=conv_lstm_kernel,
        model_path=model_path,
        learning_rate=learning_rate,
        epochs=epochs)
    model.trian(images_train, y_train, images_test, y_test, x_max, x_min, y_train_name,
                y_test_name)
