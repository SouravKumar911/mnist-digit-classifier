from tensorflow.keras.datasets import mnist
from PIL import Image

(X_train, y_train), (_, _) = mnist.load_data()

for i in range(20):
    img = X_train[i]
    label = y_train[i]

    Image.fromarray(img).save(
        f"digit_{label}_{i}.png"
    )

print("20 PNG images created successfully!")