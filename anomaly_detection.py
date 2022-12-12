from sklearn.ensemble import IsolationForest
import torch
import numpy as np
import os


if __name__ == "__main__":
    scores_path = "./iso_forest/scores"
    if os.path.exists(scores_path) is False:
        os.makedirs(scores_path)

    pred_path = "./iso_forest/predictions"
    if os.path.exists(pred_path) is False:
        os.makedirs(pred_path)
        
    mnist_train_maps = np.load("./MNIST-train/fc1-maps.npy")
    # mnist_train_maps = np.load("./MNIST-train/conv2-flattened-maps.npy")

    mnist_test_maps = np.load("./MNIST-test/fc1-maps.npy")
    # mnist_test_maps = np.load("./MNIST-test/conv2-flattened-maps.npy")
    
    notmnist_maps = np.load("./notMNIST/fc1-maps.npy")
    # notmnist_maps = np.load("./notMNIST/conv2-flattened-maps.npy")

    # combined_maps = np.load("./MNIST-notMNIST-combined/fc1-maps.npz.npy")

    clf = IsolationForest(max_samples=600, random_state=0)
    clf.fit(mnist_train_maps)

    predictions = clf.predict(mnist_test_maps)
    np.save(f"./{pred_path}/mnist-test-fc1", predictions)
    anomaly_scores = clf.score_samples(mnist_test_maps)
    np.save(f"./{scores_path}/mnist-test-fc1", anomaly_scores)

    predictions = clf.predict(notmnist_maps)
    np.save(f"./{pred_path}/notmnist-fc1", predictions)
    anomaly_scores = clf.score_samples(notmnist_maps)
    np.save(f"./{scores_path}/notmnist-fc1", anomaly_scores)
    