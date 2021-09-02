import differential_evolution_zhuru as ea
import tensorflow as tf
import numpy as np
import binascii
from itertools import groupby
import os
import perlinnoise as pn
from PIL import Image

# get feature
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


def prediction(xs, image_array_tensor, bottle_tensor, w, b, img_o):
    print('The candidate setshape:-----prediction', xs.shape)
    scores_mal = []
    fh_copy = fh

    for x in xs:

        noise = pn.NoiseUtils(length_n, 512)
        noise.makeTexture(x[0], x[1], x[2], texture=noise.wood)
        img = np.zeros((length_n, 512,3))
        pixels = img.copy()
        for i in range(0, length_n):
            for j in range(0, 512):
                for z in range(0, 3):
                    v = noise.img[i, j]
                    pixels[i, j, z] = v

        pixels = pixels.astype('float32')

        image_array_final = np.append(img_o, pixels, axis=0)

        # get feature of the image
        bottleneck_value = run_bottleneck_on_image(sess, image_array_final, image_array_tensor, bottle_tensor)

        logits = tf.matmul(bottleneck_value, w) + b

        final_tensor = tf.nn.softmax(logits)

        scores = sess.run(final_tensor)

        scores = np.squeeze(scores)
        score_mal = scores[1]
        scores_mal.append(score_mal)

    print('scores_mal len----in prediction:', len(scores_mal))
    print('best ：----', min(scores_mal))
    return scores_mal


def attack_success(score, x, fh, name):
    if score <= 0.5:
        print('success')
        noise = pn.NoiseUtils(length_n, 512)
        noise.makeTexture(x[0], x[1], x[2], texture=noise.wood)
        img = np.zeros((length_n, 512),dtype=np.int)
        pixels = img.copy()
        for i in range(0, length_n):
            for j in range(0, 512):
                v = noise.img[i, j]
                pixels[i, j] = int(v)
        fh = np.append(fh, pixels, axis=0)
        im = Image.fromarray(np.uint8(fh))
        fh = fh.flatten()
        s = ''
        for i in range(len(fh)):
            s += str(hex(fh[i])[-2:])
        s = s.replace('x', '0')
        s = binascii.unhexlify(s)
        with open('the success one.exe', 'wb') as f:
            f.write(s)
        im.show()
        im.save('save url.png')

        return True


# load exe
bytes_name = file_name('read data')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    # load model
    saver = tf.train.import_meta_graph('./checkpoint_dir/Mal_Detection_Inception.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
    print('load over')

    image_array_tensor = tf.get_default_graph().get_tensor_by_name('import/DecodeJpeg:0')

    bottle_tensor = tf.get_default_graph().get_tensor_by_name('import/pool_3/_reshape:0')

    w = tf.get_default_graph().get_tensor_by_name('final_training_ops/Variable:0')
    b = tf.get_default_graph().get_tensor_by_name('final_training_ops/Variable_1:0')
    for a in bytes_name:
        name = a.split('\\')[-1]
        print(name)
        with open(a, 'rb') as f:
            content = f.read()
            hexst = binascii.hexlify(content)
            fh = np.array([int(hexst[i:i + 2], 16) for i in range(0, len(hexst), 2)])
            print('fh', fh)
            rn = len(fh) / 512
            rn = int(rn)
            rn += 1
            print('rn:', rn)
            print('(x+1)*16 2d-array：')
            shape_new = rn * 512
            diff = shape_new - len(fh)
            for i in range(diff):
                fh = np.append(fh, 0)
            print('len after', len(fh))
            fh = np.reshape(fh[:rn * 512], (-1, 512))
            print('fh shape:', fh.shape)
            img_o = np.zeros((rn, 512, 3))
            for i in range(rn):
                for j in range(512):
                    for z in range(3):
                        img_o[i, j, z] = fh[i][j]
            img_o.astype('float32')
            length_n = int(0.3 * rn)
            bounds = [(1, 100), (1, 100), (1, 100)]
            popmul = max(1, 400 // len(bounds))
            callback_fn = lambda score, x, convergence: attack_success(score, x, fh, name)
            predict_fn = lambda x: prediction(x, image_array_tensor, bottle_tensor, w, b, img_o)
            r = ea.differential_evolution(predict_fn, bounds, maxiter=20, popsize=popmul,
                                          recombination=1, atol=-1, callback=callback_fn, disp=True, polish=False)
            # print(r.x,r.fun)


