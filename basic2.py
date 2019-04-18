import tensorflow as tf

#tf.placeholder: 계산을 실행할 때 입력값을 받는 매개변수
#tf.constant(10): 상수임
#None: 크기가 정해지지 않음

X = tf.placeholder(tf.float32, [None, 3]) # none 행 3열


print(X)

#X_san 플레이스 홀더에 넣을 값임

x_data = [[1, 2, 3], [4, 5, 6]]

#tf.Variable: 그래프를 계산 하면서 최적화 할 변수들임, 이 값이 바로 신경망을 좌우하는 값(Weight)
#tf.random_normal: 각 변수들의 초기값을 정규분포 랜덤 값으로 초기화 함

W = tf.Variable(tf.random_normal([3, 2])) # 3행 2열의 랜덤값
b = tf.Variable(tf.random_normal([2, 1])) # 2행 1열의 랜덤값

#입력값과 변수들을 계산할 수식을 작성함
#tf.matmul 처럼 mat* 로 되어 있는 함수로 행렬 계산을 수행함
expr = tf.matmul(X, W) + b
#나는 x_data를 바로 넣을것이라 생각했는데 x_data는 그저 파이썬 문법인 리스트(배열같은)일 뿐이고 실제 TF의 매개변수 역할을 하는것은 tf.placeholder

sess = tf.Session()
#위에서 설정한 Variable 들의 값들을 초기화 하기위해
#처음에 tf.global_variables_initilizer를 한 번 실행해야 함
sess.run(tf.global_variables_initializer())

print("=== x_data ===")
print(x_data)

print("=== W ===")
print(sess.run(W))

print("=== b ===")
print(sess.run(b))

print("=== expr ===")
#expr 수식에는 X 라는 입력값이 필요함
#따라서 expr 실행시에는 이 변수에 대한 실제 입력값을 다음처럼 넣어줘야 함

print(sess.run(expr, feed_dict={X: x_data}))


sess.close()

