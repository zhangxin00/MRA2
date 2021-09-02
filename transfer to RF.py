from sklearn.externals import joblib
import os
import binascii
import numpy as np
import tensorflow as tf
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.EXE' or os.path.splitext(file)[1] == '.exe':
                L.append(os.path.join(root, file))
    return L

def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})

    return bottleneck_values


# load exe
exe_name = file_name('dataset')
clf=joblib.load('RF.m')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
    for a in exe_name:
        name = a.split('/')[-1]
        print(name)
        with open(a, 'rb') as f:
            content = f.read()
            hexst = binascii.hexlify(content)
            fh = np.array([int(hexst[i:i + 2], 16) for i in range(0, len(hexst), 2)])
            print('fh', fh)
            rn = len(fh) / 512
            rn = int(rn)
            fh = np.reshape(fh[:rn * 512], (-1, 512))
            print(fh.shape)
        img_f = np.zeros((rn, 512, 3))
        for i in range(rn):
            for j in range(512):
                for z in range(3):
                    img_f[i, j, z] = fh[i][j]
        img_f.astype('float32')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('./checkpoint_dir/Mal_Detection_Inception.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
        image_array_tensor = tf.get_default_graph().get_tensor_by_name('import/DecodeJpeg:0')
        bottle_tensor = tf.get_default_graph().get_tensor_by_name('import/pool_3/_reshape:0')
        bottleneck_value = run_bottleneck_on_image(sess, img_f, image_array_tensor, bottle_tensor)
        label = clf.predict([bottleneck_value][0])
        mal_score = clf.predict_proba([bottleneck_value][0][1])