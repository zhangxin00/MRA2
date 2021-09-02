import os
import time
import sklearn.ensemble as se
import numpy as np
from sklearn.externals import joblib
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L.append(os.path.join(root, file))
    return L

def Train(train_sample,train_label):
    print(time.strftime('%Y-%m-%d %H:%M:%S'))

    rf=se.RandomForestClassifier()

    rf.fit(train_sample, train_label)

    print('over')
    return rf

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

mal_train=np.array(mal_train)
mal_label=np.array(mal_label)
print(mal_train.shape)
print(mal_label.shape)

model=Train(mal_train,mal_label)

print(model.predict_proba(mal_train[-4:]))

joblib.dump(model, "RF.m")