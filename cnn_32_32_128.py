from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D
from keras.layers import MaxPooling2D, Flatten
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical



def build_model(input_shape=(28, 28,1), num_classes=10):
    model = Sequential()

    # ------------- Your Code here ---------------
    # 1. Add the first convolution layer with 5x5 filters and a max pooling layer here 
    model.add(Conv2D(32, kernel_size=5, activation="relu",padding="same", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2))
    # 2. Add the second convolution with 3x3 filters and a max pooling layer here 
    model.add(Conv2D(32, kernel_size=3,padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3,padding="same", activation="relu"))


    model.add(Flatten())

    # 3. Add a fully connected hidden layer here
    model.add(Dense(500,activation="relu"))
    # 4. Add the final classification layer here
    model.add(Dense(num_classes,activation="softmax"))
    model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
            )

    return model


def use_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train/255
    x_test = x_test/255

    x_train_reshaped = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test_reshaped = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_train_one_hot = keras.utils.to_categorical(y_train,num_classes=10)
    y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=10)

    # Train the model you created
    my_model = build_model(input_shape=(28,28,1), num_classes=10)
    print("Training:")
    my_model.fit(
            x_train_reshaped,
            y_train_one_hot,
            epochs=1,
            batch_size=1024,
            validation_split=0.75
            )

    # Evaluate the training quality on the test dataset
    print("Inference:")
    score = my_model.evaluate(x_test_reshaped, y_test_one_hot, batch_size=128)
    print("Loss and Accuracy on test set: " + str(score))


if __name__ == '__main__':
    use_model()
