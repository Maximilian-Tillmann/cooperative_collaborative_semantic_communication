import torch
import os
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



def load_mnist(path="datasets/MNIST", seed=42):
    # set seed
    random.seed(seed)
    np.random.seed(seed)

    # load Dataser
    os.makedirs(path, exist_ok=
                True)
    
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root=path, download=True, train=True, transform=mnist_transform)
    test_dataset  = torchvision.datasets.MNIST(root=path, download=True, train=False, transform=mnist_transform)

    data_sample = train_dataset.data.view(-1,784) / 255
    # Task1: classification of 2
    task_one_targets = torch.zeros_like(train_dataset.targets)
    task_one_targets[train_dataset.targets == 2] = 1
    first_task_dataset = torch.utils.data.TensorDataset(data_sample, task_one_targets)

    # Task2: categorical classification
    # We may use the train_dataset itself
    second_task_dataset = torch.utils.data.TensorDataset(data_sample, train_dataset.targets)

    # ---->> Test set of MNIST has a different datatype from the training set, take care
    test_dataset.data = test_dataset.data.to(torch.float32)
    test_dataset.targets = test_dataset.targets.to(torch.float32)
    
    test_sample = test_dataset.data.view(-1,784) / 255
    # Accuracy Test: Task1
    first_test_target = torch.zeros_like(test_dataset.targets)
    first_test_target[test_dataset.targets == 2] = 1
    first_test_dataset = torch.utils.data.TensorDataset(test_sample, first_test_target)

    # Accuracy Test: Task2
    second_test_dataset = torch.utils.data.TensorDataset(test_sample, test_dataset.targets)

    return first_task_dataset, second_task_dataset, first_test_dataset, second_test_dataset

# Why the data has been changed to numpy?! Maybe because it is for a long time ago and hasn't been updated

def load_dataset(key):
    loader = {
        "MNIST": load_mnist,
        #"OMNIGLOT": load_omniglot,
        #"Histopathology": load_histopathology,
        #"FreyFaces": load_freyfaces,
        #"OneHot": load_one_hot
    }
    return loader[key]()

 


# Tests and veryfing


#first_task_dataset, second_task_dataset, first_test_dataset, second_test_dataset = load_dataset(key="MNIST")

#---->> (checking the training sets we have made)

# Convert the first_task_dataset to numpy arrays 
#sample_array, target_array = zip(*second_task_dataset)
#sample_array = torch.stack(sample_array).numpy()
#target_array = torch.stack(target_array).numpy()

# Now you can check the shape
#print("CommonInput shape:", sample_array.shape)
#print("Label shape:", target_array.shape)


#num_label_0 = np.sum(target_array == 0)

#print(f"Number of data points labeled 0: {num_label_0}")

# Assuming resampled_dataset is your resampled dataset
#N = len(second_task_dataset)

#print(f"Number of samples in the first task dataset: {N}")


# ---->> (Ploting a recovered data)

#index_2 = next(i for i, (image, label) in enumerate(first_task_dataset) if label == 1)

# Extract and plot the image
#image_2, label = first_task_dataset[index_2]
#image_2 = image_2.numpy()  # Convert to NumPy array
#image_2 = image_2.reshape(28, 28)  # Reshape to the original 2D shape
#plt.imshow(image_2, cmap='gray')
#plt.title(f'MNIST Image labeled as {label}')
#plt.show()


# ---->> (Testing the way we use the test dataset reshaping in the eval loop)

#test_reshaped = test_dataset.data.view(-1,784).squeeze()

#neu_test_dataset = torch.utils.data.TensorDataset(test_reshaped, test_dataset.targets)


#index_5 = next(i for i, (image, label) in enumerate(second_test_dataset) if label == 8)

# Extract and plot the image
#image_5, label = second_test_dataset[index_5]
#image_5 = image_5.numpy()  # Convert to NumPy array
#image_5 = image_5.reshape(28, 28)  # Reshape to the original 2D shape
#plt.imshow(image_5, cmap='gray')
#plt.title(f'MNIST Image labeled as {label}')
#plt.show()

# Assuming your datasets are named first_task_dataset, second_task_dataset, first_test_dataset, and second_test_dataset

# Check data type of training datasets
#print("First Task Training Dataset Data Type:", first_task_dataset.tensors[0].dtype)
#print("Second Task Training Dataset Data Type:", second_task_dataset.tensors[0].dtype)

# Check data type of testing datasets
#print("First Task Test Dataset Data Type:", first_test_dataset.tensors[0].dtype)
#print("Second Task Test Dataset Data Type:", second_test_dataset.tensors[0].dtype)
