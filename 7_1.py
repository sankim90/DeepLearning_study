import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # 28 x 28 입력
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) # 1채널(gray scale)3 x 3 컨볼루션 필터 32개 4차원 배열 맞음
# W1 [3 3 1 32] -> [3 3]: 커널 크기, 1: 입력값 X 의 특성수, 32: 필터 갯수


             #  a  b  c  d
t = np.array(# (3, 3, 3, 4)
[

    [#a1      #c1          #c2           #c3
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],#b1

        [[4, 5, 6, 7], [1, 2, 3, 4], [1, 2, 3, 4]],#b2

        [[7, 8, 9, 10], [1, 2, 3, 4], [1, 2, 3, 4]]#b3
         #d1 d2 d4 d4
    ],

    [#a2
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],

        [[4, 5, 6, 7], [1, 2, 3, 4], [1, 2, 3, 4]],

        [[7, 8, 9, 10], [1, 2, 3, 4], [1, 2, 3, 4]]
    ],

    [#a3
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],

        [[4, 5, 6, 7], [1, 2, 3, 4], [1, 2, 3, 4]],

        [[7, 8, 9, 10], [1, 2, 3, 4], [1, 2, 3, 4]]
    ]
]
)

print(t.shape)

#tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
#설명: Outputs random values from a normal distribution.

#Args:
#shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
#mean: A 0-D Tensor or Python value of type dtype. The mean of the normal distribution.
#stddev: A 0-D Tensor or Python value of type dtype. The dstandard deviation of the normal distribution.
#dtype: The type of the output.
#seed: A Python integer. Used to create a random seed for the distribution. See set_random_seed for behavior.
#name: A name for the operation (optional).

L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
#tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)

#input : [batch, in_height, in_width, in_channels] 형식. 28x28x1 형식의 손글씨 이미지.
#filter : [filter_height, filter_width, in_channels, out_channels] 형식. 3, 3, 1, 32의 w.
#strides : 크기 4인 1차원 리스트. [0], [3]은 반드시 1. 일반적으로 [1], [2]는 같은 값 사용.
#padding : 'SAME' 또는 'VALID'. 패딩을 추가하는 공식의 차이. SAME은 출력 크기를 입력과 같게 유지.

L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 2 x 2 풀링 14 x 14
#tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)

#value : [batch, height, width, channels] 형식의 입력 데이터. ReLU를 통과한 출력 결과가 된다.
#ksize : 4개 이상의 크기를 갖는 리스트로 입력 데이터의 각 차원의 윈도우 크기.
#ksize가 [1,2,2,1]이라는 뜻은 2칸씩 이동하면서 출력 결과를 1개 만들어 낸다는 것이다. 다시 말해 4개의 데이터 중에서 가장 큰 1개를 반환하는 역할을 한다.
#data_format : NHWC 또는 NCHW. n-count, height, width, channel의 약자 사용.

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 2 x 2 풀링 7 x 7

W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

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
            feed_dict ={X: batch_xs, Y: batch_ys, keep_prob: 0.7})

        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

print('Optimize done')


# 결과 확인

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                            Y: mnist.test.labels,
                                            keep_prob: 1}))