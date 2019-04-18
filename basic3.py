import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#tf.random_uniform 함수는 정규 분포 난수를 생성하는 함수로, 배열의 shape, 최소값 최대값을 파라미터로 사용함
#여기서 [1], -1.0, 1.0을 전달하기 때문에 -1 ~ 1사이의 난수 1개를 만듬, 5개만들고 싶으면 [5]

#name: 나중에 텐서보드등로 값의 변화를 추적하거나 살펴고기 쉽게 하기 위해 이름을 붙임

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
print(X)
print(Y)

#X와 Y의 상관 관계를 분석하기 위한 가설(H(x))를 작성함
#y = W * x + b
#W 와 X 가 행렬이 아니므로 tf.matmul이 아니라 기본 곱셈 기호를 사용함
hypothesis = W * X + b

#loss function 작성
#mean(h - Y)^2

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 텐서플로우에 기본적으로 포함되어 있는 함수를 이용해 경사 하강법 최적화를 수행함
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

#비용을 최소화 하는것이 최종 목표
train_op = optimizer.minimize(cost)

#세션을 생성하고 초기화 함
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #최적화를 100번 수행함

    for step in range(100):
        #sess.run을 통해 train_op와 cost그래프를 계산 함
        #이때, H(x)에 넣어야 할 실제값을 feed_dict를 통해 전달 함
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        print(cost_val, sess.run(W), sess.run(b))

#최적화 완료?
    print("\n === Test ====")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X:2.5, y:", sess.run(hypothesis, feed_dict={X: 2.5}))