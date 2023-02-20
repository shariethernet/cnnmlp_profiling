from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Conv2D
from keras.layers import MaxPooling2D, Flatten
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import h5py


model = load_model("cnn_32_32_128_mnist.h5")


def use_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #x_train = x_train/255
    x_test = x_test/255

    #x_train_reshaped = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test_reshaped = x_test.reshape(x_test.shape[0], 28, 28, 1)
    #y_train_one_hot = keras.utils.to_categorical(y_train,num_classes=10)
    y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=10)

    # Train the model you created
    
    # Evaluate the training quality on the test dataset
    # print("Inference:")
    score = model.evaluate(x_test_reshaped, y_test_one_hot, batch_size=128)
    print("Loss and Accuracy on test set: " + str(score))


if __name__ == '__main__':
    use_model()
