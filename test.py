from tensorflow.keras.datasets import mnist
from PIL import Image

(X_train, y_train), (_, _) = mnist.load_data()

Image.fromarray(X_train[0]).save("test_digit.png")

print("Actual Digit:", y_train[0])