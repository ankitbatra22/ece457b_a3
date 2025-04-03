from src.dataset_utils import load_fashion_mnist_twist

x_train, y_train, x_test, y_test = load_fashion_mnist_twist(data_path='./data')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)