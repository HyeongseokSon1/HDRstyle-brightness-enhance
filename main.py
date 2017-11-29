import tensorflow as tf
import numpy as np
import PIL.Image
import shutil, os
import dataload
import scipy.misc
from argparse import ArgumentParser

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--mode',
        dest='mode',required=True)    
    parser.add_argument('--input',
        dest='input',required=True)
    return parser

parser = build_parser()
options = parser.parse_args()

mode = options.mode
batch_size = 16
x_pattern = '/home/sonhs/data1/5K_small/expertC/*'
y1_pattern = '/home/sonhs/data1/5K_small/results/*'
y2_pattern = '/home/sonhs/data1/5K_small/expertC/*'
# end_idx = 1016
num_epoch = 10

lr_init = 0.0001
training_iters = 100001
display_step = 100
beta1 = 0.9
lr_decay = 0.9
lr_decay_step = 2500

if mode == 'train':
    streamer1 = dataload.FileStreamer('x', x_pattern)#, end_idx=end_idx)
    streamer2 = dataload.FileStreamer('y1', y1_pattern)#, end_idx=end_idx)
    streamer3 = dataload.FileStreamer('y2', y2_pattern)#, end_idx=end_idx)
    streamer1 = dataload.ImageStreamer(streamer1, shape=dataload.Coordinate(None, None, 3))
    streamer2 = dataload.ImageStreamer(streamer2, shape=dataload.Coordinate(None, None, 3))
    streamer3 = dataload.ImageStreamer(streamer3, shape=dataload.Coordinate(None, None, 3))
    streamer = dataload.JoinStreamer([streamer1, streamer2, streamer3])
    streamer = dataload.RandomCropStreamer(streamer, dataload.Coordinate(224, 224), num_crop=4)
    streamer = dataload.ParallelDataset(streamer, batch_size=batch_size)
    xx, yy, zz = streamer.ops
    h_ = 224
    w_ = 224
    c_ = 3


if mode == 'test':    
    img = PIL.Image.open(options.input)
    batch_size = 1
    w_, h_ = img.size
    #h_ = h_ - (h_%32)
    #w_ = w_ - (w_%32)
    c_ = 3;
    xx = tf.placeholder(tf.float32, [batch_size, h_, w_, c_])
    yy = tf.placeholder(tf.float32, [batch_size, h_, w_, c_])
    zz = tf.placeholder(tf.float32, [batch_size, h_, w_, c_])

def conv2d(input, output_dim, kh=3, kw=3, sh=1, sw=1, stddev=0.01, name='conv'):
    with tf.variable_scope(name):
        # Conv2D wrapper, with bias and relu activation
        w = tf.get_variable('w', [kh,kw,input.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, sh, sw, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv

def deconv2d(input, output_shape, kh=3, kw=3, sh=2, sw=2, stddev=0.01, name='deconv'):
    with tf.variable_scope(name):
        # Conv2D wrapper, with bias and relu activation
        w = tf.get_variable('w', [kh,kw, output_shape[-1], input.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=[1, sh, sw, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return deconv


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')




## Create model

def generator(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1 = conv2d(x, 32, 3, 3, name='g_conv1')
    conv2 = conv2d(tf.nn.relu(conv1), 64, 3, 3, name='g_conv2')
    conv3 = conv2d(tf.nn.relu(conv2), 64, 3, 3, name='g_conv3')
    conv4 = conv2d(tf.nn.relu(conv3), 64, 3, 3, name='g_conv4')
        ##################
    conv5_1 = conv2d(tf.nn.relu(conv4), 64, 3, 3, name='g_conv5_1')
    conv6_1 = conv2d(tf.nn.relu(conv5_1), 64, 3, 3, name='g_conv6_1')
    conv7_1 = conv2d(tf.nn.relu(conv6_1), 64, 3, 3, name='g_conv7_1')
    conv8_1 = conv2d(conv7_1, 3, 1, 1, name='g_conv8_1')
        ##################
    conv5_2 = conv2d(tf.nn.relu(conv4), 64, 1, 1, name='g_conv5_2')
    conv6_2 = conv2d(tf.nn.relu(conv5_2), 64, 1, 1, name='g_conv6_2')
    conv7_2 = conv2d(tf.nn.relu(conv6_2), 64, 1, 1, name='g_conv7_2')
    conv8_2 = conv2d(conv7_2, 3, 1, 1, name='g_conv8_2')

    result_smooth = tf.multiply(x,1+conv8_2)
    result = tf.add(result_smooth, conv8_1)
    return result_smooth, result



def discriminator(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1 = conv2d(x, 32, 3, 3, name='d_conv1')
    max1 = maxpool2d(tf.nn.relu(conv1),4)
    conv2 = conv2d(max1, 64, 3, 3, name='d_conv2')
    max2 = maxpool2d(tf.nn.relu(conv2),4)
    conv3 = conv2d(max2, 128, 3, 3, name='d_conv3')
    max3 = maxpool2d(tf.nn.relu(conv3),4)
    conv4 = conv2d(max3, 1024, 3, 3, name='d_conv4')
    max4 = maxpool2d(tf.nn.relu(conv4),4)
    conv5 = conv2d(max4, 1, 1, 1, name='d_conv5')
    return conv5


def styleloss(input1, input2):
    input1 = input1
    input2 = input2
    b = input1.get_shape()[0].value
    h = input1.get_shape()[1].value
    w = input1.get_shape()[2].value
    c = input1.get_shape()[3].value

    outputs = []
    for i in range(b):
        input1_batch = tf.slice(input1, [i, 0, 0, 0], [1, -1, -1, -1])
        input1_trans = tf.transpose(input1_batch, [0, 3, 1, 2])
        input1_reshape = tf.reshape(input1_trans,[c, h*w])
        Gram1 = tf.matmul(input1_reshape, tf.transpose(input1_reshape))
        input2_batch = tf.slice(input2, [i, 0, 0, 0], [1, -1, -1, -1])
        input2_trans = tf.transpose(input2_batch, [0, 3, 1, 2])
        input2_reshape = tf.reshape(input2_trans, [c, h * w])
        Gram2 = tf.matmul(input2_reshape, tf.transpose(input2_reshape))
        loss_batch = tf.scalar_mul(1./(h*w*c),tf.reduce_sum(tf.square(tf.subtract(Gram1, Gram2))))
        outputs.append(loss_batch)

    outputs = tf.stack(outputs)
    return outputs

input_hist = tf.summary.image("xx",xx)
label_hist = tf.summary.image("yy",yy)

zeros = tf.fill([batch_size, 1, 1, 1], 0.)
ones = tf.fill([batch_size, 1, 1, 1], 1.)

# yy = xx*0.5 + yy*0.5

with tf.variable_scope('model'):
    Gsmooth, G = generator(xx, reuse=False)
    D = discriminator(G,reuse=False)
    D_ = discriminator(zz,reuse=True)
d_cost_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_,labels=tf.ones_like(D_)))
d_cost_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D,labels=tf.zeros_like(D)))
g_cost = 40e-2 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D,labels=tf.ones_like(D))) #+ 0.0001*sum(weights_regul) # + 0.0000001*(tf.nn.l2_loss(weights['wc1']) + tf.nn.l2_loss(weights['wc2']) + tf.nn.l2_loss(weights['wc3']))
d_cost = 10e-1 * (d_cost_real + d_cost_fake)

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

output_hist = tf.summary.image("res",G)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./log")
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    if mode == 'train':
        e_cost = tf.scalar_mul(1. / (3 * 224 * 224), tf.reduce_sum(tf.square(tf.subtract(yy, G))))        
        r_cost = tf.scalar_mul(1. / (3 * 56 * 56), tf.reduce_sum(tf.square(tf.subtract(tf.image.resize_bilinear(yy,[56,56]), tf.image.resize_bilinear(G,[56,56])))))
        g_cost = g_cost + r_cost

        lr_v = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr_init, lr_v, lr_decay_step, lr_decay, staircase=True)
        e_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(e_cost, var_list=(g_vars), global_step=lr_v)
        r_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(r_cost, var_list=(g_vars), global_step=lr_v)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_cost, var_list=(g_vars), global_step=lr_v)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(d_cost, var_list=(d_vars), global_step=lr_v)

        ## Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)
        ## Keep training until reach max iterations
        streamer.start(sess)

        for iterations in range(training_iters):
            # Run optimization op (backprop)
            if iterations <= 5000:
                sess.run(e_optimizer)
            else:
                sess.run(g_optimizer)
                sess.run(d_optimizer)

            if iterations % display_step == 0:
                ## Calculate batch loss and accuracy
                loss1, loss2, loss3, loss4, summary_str, lr_value = sess.run([e_cost, r_cost, g_cost, d_cost, merged, learning_rate])
                writer.add_summary(summary_str, iterations)
                print("Iter " + str(iterations) + ", Minibatch Loss= " + \
                      str(loss1) + "(e_loss), " + str(loss2) + "(r_loss), " + str(loss3) + "(g_loss), " + str(loss4) + "(d_loss), " + str(lr_value)+"(lr_value)")

            if iterations % (display_step * 5) == display_step - 1:
                xxx, yyy, result = sess.run([xx, yy, G])
                bx = np.reshape(xxx, (batch_size * h_, w_, c_))
                result = np.reshape(result, (batch_size * h_, w_, c_))
                by = np.reshape(yyy, (batch_size * h_, w_, c_))
                scipy.misc.toimage(np.concatenate((bx, result, by), axis=1), cmin=0.0, cmax=1.0).save("result_tmp.png")
                print('displayed')

            if iterations % 10000 == 0:
                save_path = saver.save(sess, "tmp/model",global_step=iterations)

        print("Optimization Finished!")

    if mode == 'test':
        saver.restore(sess,"model/model-50000")
        img = np.asarray(img,'float') * (1./255)
        result_img = sess.run(G, feed_dict={xx: [img], yy: [img]})
        result_img = np.reshape(result_img, (h_,w_,3))
        scipy.misc.toimage(result_img, cmin=0.0, cmax=1.0).save("result_img.png")




