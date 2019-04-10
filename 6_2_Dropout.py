import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784]) # X input의 batch에 따라 자동으로 계산하게끔 None
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32) # 학습시는 뉴런 drop out, 인퍼런스시 뉴런 모두 사용

W1 = tf.Variable(tf.random_normal([784, 256], stddev = 0.01)) #stddev 표준편차가 0.01인 정규분포를 가지는 임의의 값으로 뉴런을 초기화
b1 = tf.Variable(tf.zeros([256])) #zeros [None, 256] 으로 했다가 에러: 열벡터로만 만들면 되는것 같음

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L1 = tf.nn.dropout(L1, keep_prob) # tf.nn.dropout(L1, 0.8): 학습시 80%의 뉴런만 사용하겠다.

W2 = tf.Variable(tf.random_normal([256, 256], stddev = 0.01))
b2 = tf.Variable(tf.zeros([256]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev = 0.01))
b3 = tf.zeros([10])

model = tf.add(tf.matmul(L2, W3), b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost) # 0.001 아마도 learing rete 인듯?

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost],
            feed_dict ={X: batch_xs, Y: batch_ys, keep_prob: 0.8})

        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

print('Optimize done')


# 결과 확인

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
#print('정확도:', sess.run(accuracy, feed_dict={X: mnist.train.images, Y: mnist.train.labels, keep_prob: 1}))

#print(W3)
#print(model)
#print(sess.run(model, feed_dict={X: mnist.test.images, Y: mnist.test.labels})[0])

#print(is_correct)
#print(sess.run(is_correct, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))