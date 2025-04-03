import pandas as pd
import numpy as np
import os

def load_fashion_mnist_twist(data_path='../data'):
    train_features_path = os.path.join(data_path, 'x_train.csv')
    train_labels_path = os.path.join(data_path, 'y_train.csv')
    test_features_path = os.path.join(data_path, 'x_test.csv')
    test_labels_path = os.path.join(data_path, 'y_test.csv')

    try:
        print("Loading data, skipping header row...")
        x_train_df = pd.read_csv(train_features_path, header=None, skiprows=1) # Added skiprows=1
        y_train_df = pd.read_csv(train_labels_path, header=None, skiprows=1)   # Added skiprows=1
        x_test_df = pd.read_csv(test_features_path, header=None, skiprows=1)    # Added skiprows=1
        y_test_df = pd.read_csv(test_labels_path, header=None, skiprows=1)     # Added skiprows=1

        print("Inspecting the dataframes after skipping header")
        print("x_train_df info: ")
        x_train_df.info() 
        print("\ny_train_df info: ")
        y_train_df.info()
        print("\nx_test_df info: ")
        x_test_df.info()
        print("\ny_test_df info: ")
        y_test_df.info()

        # print("x_train_df describe: ", "\n", x_train_df.describe())
        # print("y_train_df describe: ", "\n", y_train_df.describe())
        # print("x_test_df describe: ", "\n", x_test_df.describe())
        # print("y_test_df describe: ", "\n", y_test_df.describe())


        x_train = x_train_df.values
        y_train = y_train_df.values.ravel() # Flatten label array
        x_test = x_test_df.values
        y_test = y_test_df.values.ravel()   # Flatten label array

        print(f"\nData loaded successfully:")
        print(f"  x_train shape: {x_train.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  x_test shape: {x_test.shape}")
        print(f"  y_test shape: {y_test.shape}")

        expected_train_samples = 60000
        expected_test_samples = 10000
        if x_train.shape[0] == expected_train_samples and \
           y_train.shape[0] == expected_train_samples and \
           x_test.shape[0] == expected_test_samples and \
           y_test.shape[0] == expected_test_samples:
            print("Sample counts match expected values (60k train, 10k test).")
        else:
            print("Warning: Sample counts DO NOT match expected values!")

        return x_train, y_train, x_test, y_test

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Please ensure the following files exist in the '{data_path}' directory:")
        print(f"  - x_train.csv")
        print(f"  - y_train.csv")
        print(f"  - x_test.csv")
        print(f"  - y_test.csv")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None, None, None, None



def preprocess_data(x_train, x_test, num_classes=5):

    if x_train is None or x_test is None:
        print("Error: Input data (x_train or x_test) is None.")
        return None, None

    img_rows, img_cols = 28, 28
    num_channels = 1
    try:
        # Reshape to (N, H, W) first, then add channel dimension at axis 1 -> (N, C, H, W)
        x_train_reshaped = x_train.reshape(x_train.shape[0], img_rows, img_cols)
        x_test_reshaped = x_test.reshape(x_test.shape[0], img_rows, img_cols)

        x_train_reshaped = np.expand_dims(x_train_reshaped, axis=1)
        x_test_reshaped = np.expand_dims(x_test_reshaped, axis=1)

        # Convert data type to float32 and normalize pixel values to [0, 1]
        x_train_processed = x_train_reshaped.astype('float32') / 255.0
        x_test_processed = x_test_reshaped.astype('float32') / 255.0

        print("Data preprocessing completed (PyTorch NCHW format):")
        print(f"  x_train reshaped: {x_train_processed.shape}") # Should be (N, 1, 28, 28)
        print(f"  x_test reshaped: {x_test_processed.shape}")   # Should be (N, 1, 28, 28)
        print(f"  Pixel values normalized.")

        return x_train_processed, x_test_processed

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return None, None

if __name__ == '__main__':
    print("Testing data utilities...")
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_fashion_mnist_twist(data_path='../data')

    if x_train_raw is not None:
        print("\nRaw data shapes:")
        print(f"x_train: {x_train_raw.shape}, y_train: {y_train_raw.shape}")
        print(f"x_test: {x_test_raw.shape}, y_test: {y_test_raw.shape}")
        print(f"Unique labels in y_train: {np.unique(y_train_raw)}")

        x_train_proc, x_test_proc = preprocess_data(x_train_raw, x_test_raw)

        if x_train_proc is not None:
            print("\nProcessed data shapes:")
            print(f"x_train_processed: {x_train_proc.shape}")
            print(f"x_test_processed: {x_test_proc.shape}")
            print(f"Min/Max pixel values in processed train data: {x_train_proc.min()}, {x_train_proc.max()}")