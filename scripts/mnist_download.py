from torchvision import datasets

if __name__ == "__main__":
    print("Downloading MNIST dataset...")
    data = datasets.mnist.MNIST("./",download = True)
