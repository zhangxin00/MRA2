import differential_evolution as ea
import tensorflow as tf
import numpy as np
import binascii
import os
from PIL import Image


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


def prediction(xs, image_array_tensor, bottle_tensor, w, b, fh,diff):
    print('The candidate set shape:-----prediction', xs.shape)
    scores_mal = []
    fh_copy = fh

    for x in xs:
        candicates=[[x[i],x[i+1],x[i+2]] for i in range(0,len(x),3)]
        for cadicate in candicates:
            a=int(cadicate[0])
            row=report[a]
            fh_copy[row, int(cadicate[1])] = int(cadicate[2])

        fh_one = fh_copy.flatten()

        fh_one = np.delete(fh_one, np.s_[-diff:])

        width_image = 512
        rn = len(fh_one) / width_image
        rn = int(rn)

        array = np.reshape(fh_one[:rn * width_image], (-1, width_image))
        # array=np.uint8(array)

        img_f = np.zeros((rn, width_image, 3))
        for i in range(rn):
            for j in range(width_image):
                for z in range(3):
                    img_f[i, j, z] = array[i][j]
        img_f.astype('float32')


        bottleneck_value = run_bottleneck_on_image(sess, img_f, image_array_tensor, bottle_tensor)

        logits = tf.matmul(bottleneck_value, w) + b

        final_tensor = tf.nn.softmax(logits)

        scores = sess.run(final_tensor)

        scores = np.squeeze(scores)
        score_mal = scores[1]
        scores_mal.append(score_mal)

    print('scores_mal len----in prediction:', len(scores_mal))
    print('best in the current', min(scores_mal))
    return scores_mal



def attack_success(score, x, fh, name, diff):
    if score <= 0.5:
        print('success')
        candicates = [[x[i], x[i + 1], x[i + 2]] for i in range(0, len(x), 3)]
        for cadicate in candicates:
            a = int(cadicate[0])
            row = report[a]
            fh[row, int(cadicate[1])] = int(cadicate[2])

        fh = fh.flatten()
        fh = np.delete(fh, np.s_[-diff:])
        s = ''
        for i in range(len(fh)):
            s += str(hex(fh[i])[-2:])
        s = s.replace('x', '0')
        s = binascii.unhexlify(s)
        with open('the success one.exe', 'wb') as f:
            f.write(s)
        width_image = 512
        rn = len(fh) / width_image
        rn = int(rn)
        array = np.reshape(fh[:rn * width_image], (-1, width_image))
        im = Image.fromarray(np.uint8(array))
        im.show()
        im.save('save url.png')

        return True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph('./checkpoint_dir/Mal_Detection_Inception.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
    print('load  over')

    image_array_tensor = tf.get_default_graph().get_tensor_by_name('import/DecodeJpeg:0')

    bottle_tensor = tf.get_default_graph().get_tensor_by_name('import/pool_3/_reshape:0')

    w = tf.get_default_graph().get_tensor_by_name('final_training_ops/Variable:0')
    b = tf.get_default_graph().get_tensor_by_name('final_training_ops/Variable_1:0')

    bytes_name = file_name('read data')
    for a in bytes_name:
        name = a.split('\\')[-1]
        print(name)
        with open(a, 'rb') as f:
            content = f.read()
            hexst = binascii.hexlify(content)
            fh = np.array([int(hexst[i:i + 2], 16) for i in range(0, len(hexst), 2)])
            print('fh', fh)
            rn = len(fh) / 16
            rn = int(rn)
            rn += 1
            print('rn:', rn)

            shape_new = rn * 16
            diff = shape_new - len(fh)
            for i in range(diff):
                fh = np.append(fh, 1)
            print('len after', len(fh))
            fh = np.reshape(fh[:rn * 16], (-1, 16))
            print(fh.shape)

            report = []
            for i in range(len(fh) - 1):
                if (fh[i] == fh[i + 1]).all() == True:
                    report.append(i)
            print('len(report):', len(report))

            bounds = [(0, len(report)), (0, 16), (0, 256)] * 10
            popmul = max(1, 500 // len(bounds))
            callback_fn = lambda score, x, convergence: attack_success(score, x, fh, name, diff)
            predict_fn = lambda x: prediction(x, image_array_tensor, bottle_tensor, w, b, fh, diff)
            r = ea.differential_evolution(predict_fn, bounds, maxiter=25, popsize=popmul,
                                          recombination=1, atol=-1, callback=callback_fn, disp=True, polish=False)

            if 'session' in locals() and tf.sess is not None:
                print('Close interactive session')
                tf.sess.close()



