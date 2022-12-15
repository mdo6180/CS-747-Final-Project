# CS-747-Final-Project

Instructions to setup project

$ cd CS-747-Final-Project

Download notMNIST dataset from here (https://www.kaggle.com/datasets/lubaroli/notmnist)
Download the MNIST dataset from here (http://yann.lecun.com/exdb/mnist/)
Note: all the data should be inside the CS-747-Final-Project folder

Install pip
```
$ python -m ensurepip --upgrade
```

Install pipenv
```
$ pip install pipenv
```

Activate pipenv
```
$ pipenv shell
```

Run train_model.py to train the LeNet5 model
```
$ python3 train_model.py
```

Run extract_maps.py to extract activation maps
```
$ python3 extract_maps.py
```

Run anomaly_detection.py to get the anomaly scores
```
$ python3 anomaly_detection.py
```

Run the cells in the analysis.ipynb jupyter notebook to compute the metrics

Disclaimer: 
1. You may have to make some changes to the each file (may just be some paths that are local to my machine).
2. I ran these files on an AWS g4dn.xlarge instance using this AMI (https://aws.amazon.com/marketplace/pp/prodview-64e4rx3h733ru?sr=0-3&ref_=beagle&applicationId=AWSMPContessa)

