import tensorflow as tf
import numpy as np

#[털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]
)

# [기타, 포유류, 조류]
# 다음과 같은 형식을 one-hot 형식의 데이터라고 함
y_data = np.array([
    [1, 0, 0], #기타
    [0, 1, 0], #포유류
    [0, 0, 1], #조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])


# 신경망 모델 구성

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 신경망은 2차원으로 [입력층(특성), 출력층(레이블)] -> [2, 3] 으로 정함
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.)) # input x data 2, hidden layer output 10
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.)) # input hidden layer 10, output classify 3
#hidden layer 뉴런 수는 hyper param으로 실험을 통해 정해짐

# bias을 각각의 레이어의 아웃풋 갯수로 설정함
# bias는 아웃풋 갯수, 즉 최종된 결과값의 수인 3으로 설정함
b1 = tf.Variable(tf.zeros([10])) # hidden layer 10
b2 = tf.Variable(tf.zeros([3]))  # classify 3

L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

model = tf.add(tf.matmul(L1, W2), b2)
#보통 출력층에서는 활성화 함수 적용 안함
#여기까지가 NN 모델

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

#cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis = 1))
#크로스 엔트로피 손실 함수 파일 참고(크롬 즐겨찾기, tensorflow study 폴더)
# reduce_* 설명: 텐서의 차원을 줄임
# _* : 축소 방법
# axis: 축소할 차원

# Y * tf.log(model)    reduce_sum(axis=1)
#[[ -1.0, 0, 0      -> [-1.0, -0.09]
#   0, -0.09, 0]]

#          reduce_mean
# [-1.0, -0.09] -> -0.545

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

#텐서플로우 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 학습 100 번
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    #학습 도중 10번에 한 번씩 손실값을 출력
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))


prediction = tf.argmax(model, axis=1) #argmax: argumnet 중에 가장 큰 값의 인덱스를 return
target = tf.argmax(Y, axis=1)

print('예측값:', sess.run(prediction, feed_dict={X: x_data})) ##결과 값은 [기타, 포유류, 조류]의 인덱스 0, 1, 2
print('실제값:', sess.run(target, feed_dict={Y: y_data})) #y_data 보고 정답의 인덱스를 출력

is_correct = tf.equal(prediction, target)
#print(sess.run(is_correct, feed_dict={X: x_data, Y: y_data}))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#print(sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

