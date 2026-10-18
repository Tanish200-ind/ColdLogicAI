
import urllib.request
from pathlib import Path
import numpy as np

saved_model = np.load("models/digit_model.npz")

weight = saved_model["weights"]
bias = saved_model["biases"]

print("Saved weights loaded!")
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

file_path = data_folder / "mnist.npz"

print("Downloading MNIST dataset...")

urllib.request.urlretrieve(url, file_path)

print("Download complete!")
print("Saved to:", file_path)
data = np.load(file_path)
print(data.files)
data = np.load(file_path)

images = data["x_train"]
labels = data["y_train"]

image = images[0]
image.show()


flattened_image = image.flatten()


normalized_image = flattened_image / 255.0



output=np.dot(weight,normalized_image)+bias
index=np.argmax(output)

print("Number is(pridected)", index)
print("what is was",labels[0])


model_folder = Path("models")
model_folder.mkdir(exist_ok=True)
n=0.1



np.savez(
    model_folder / "digit_model.npz",
    weights=weight,
    biases=bias
)

print("Weights and biases saved!")