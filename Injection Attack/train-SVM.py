from sklearn.externals import joblib
import numpy as np
from sklearn import svm
import time
import os


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L.append(os.path.join(root, file))
    return L

def Train(train_sample,train_label,C,kernel,cache_size,gamma,probability):
    print(time.strftime('%Y-%m-%d %H:%M:%S'))

    clf = svm.SVC(C=C, kernel=kernel,cache_size=cache_size,gamma=gamma,probability=probability)

    clf.fit(train_sample, train_label)
    # test

    print('over')
    return clf


mal_name=file_name('malware dataset')
good_name=file_name('good dataset')
mal_label=[]
mal_train=[]

for i in mal_name:
    with open(i, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        mal_train.append(bottleneck_values)
        mal_label.append(1)
for j in good_name:
    with open(j, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        mal_train.append(bottleneck_values)
        mal_label.append(0)


# mal_train/=256.0

mal_train=np.array(mal_train)
mal_label=np.array(mal_label)
print(mal_train.shape)
print(mal_label.shape)

model=Train(mal_train,mal_label,130,'linear',800,'auto',True)

print(model.predict(mal_train[-4:]))

joblib.dump(model, "train_model.m")
