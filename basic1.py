import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
print(hello)

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b) # a + b로 대체 가능

print(c)

sess = tf.Session() #실행을 위한 객체 생성

print(sess.run(hello))
print(sess.run([a, b, c]))

sess.close()
