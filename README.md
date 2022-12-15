# CS-747-Final-Project

Instructions to setup project

$ cd CS-747-Final-Project

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
