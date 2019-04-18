import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

total_epoch = 100
batch_size = 100
n_hidden = 256
n_input = 28 * 28
n_noise = 128
n_class = 10

X = tf.placeholder(tf.float32, [None, n_input], name='x0')
Y = tf.placeholder(tf.float32, [None, n_class], name='y0')
Z = tf.placeholder(tf.float32, [None, n_noise], name='z0')

def generator(noise, labels):
    with tf.variable_scope('generator'):
        inputs = tf.concat([noise, labels], 1, name='x1')

        hidden = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu, name='conv1')

        output = tf.layers.dense(hidden, n_input, activation=tf.nn.sigmoid, name='y1')

    return output

def discriminator(inputs, labels, reuse=None):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        inputs = tf.concat([inputs, labels], 1, name='x2')

        hidden = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu, name='conv2')

        output = tf.layers.dense(hidden, 1, activation=None, name='y2')

    return output

def get_noise(batch_size, n_noise):
    return np.random.uniform(-1., 1., size=[batch_size, n_noise])

G = generator(Z, Y)
D_real = discriminator(X, Y)
D_gene = discriminator(G, Y, True)


loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))) #tf.ones_like(x) : 입력된 shape의 크기로 값을 1로 채워 만들어줌
# 활성화 함수로 sigmoid, loss 함수로 cross_entropy, binary classification은 위해 logistic regression 사용. multi classification은 softmax 사용
loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.zeros_like(D_gene))) #tf.zeros_like(x) : 입력된 shape의 크기로 값을 0으로 채워 만들어줌

loss_D = loss_D_real + loss_D_gene

loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.ones_like(D_gene)))

vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

train_D = tf.train.AdamOptimizer().minimize(loss_D, var_list=vars_D)

train_G = tf.train.AdamOptimizer().minimize(loss_G, var_list=vars_G)

sess = tf.Session()

saver = tf.train.Saver()

#sess.run(tf.global_variables_initializer())
saver.restore(sess, tf.train.latest_checkpoint('./model')) # ckpt의 파라미터로 초기화

total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

#print(mnist.test.labels[0]) # 0에서 9 사이의 one-hot encoding 된 라벨값
#print(mnist.test.images[0]) # 28 x 28 image의 RGB를 0 ~ 1 사이의 값으로 normalization 한 값
for epoch in range(total_epoch):
    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G,
                           feed_dict={Y: mnist.test.labels[:sample_size],
                                      Z: noise})

        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples2/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        print('Process samples2/{}.png..'.format(str(epoch).zfill(3)))
        plt.close(fig)
print('인퍼런스 완료!')
