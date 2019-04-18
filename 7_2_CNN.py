#고수준 API 사용

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # 28 x 28 입력
Y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)

L1 = tf.layers.conv2d(X, 32, [3,3], activation=tf.nn.relu, padding='SAME')
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2], padding='SAME')
L1 = tf.layers.dropout(L1, 0.7, is_training)

L2 = tf.layers.conv2d(L1, 64, [3, 3])
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
L2 = tf.layers.dropout(L2, 0.7, is_training)

L3 = tf.contrib.layers.flatten(L2)
L3 = tf.layers.dense(L3, 256, activation=tf.nn.relu)
L3 = tf.layers.dropout(L3, 0.5, is_training)

model = tf.layers.dense(L3, 10, activation=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
#optimizer = tf. train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
#####
#신경망 모델 학습

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost],
            feed_dict ={X: batch_xs, Y: batch_ys, is_training: True})

        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

print('Optimize done')


# 결과 확인

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                            Y: mnist.test.labels,
                                            is_training: False}))
