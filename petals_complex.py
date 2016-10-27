###############################################
#
# This program was made to solve the puzzle
#   known as "petals around the rose".
#
# Those who come to understand the puzzle 
#   become a member to the sacred organization
#   known as the potentates of the rose.
#
# Once you become a potentate, it is very
#   important that you do not share how you
#   solved the puzzle with anyone.
#
# I did this in an attempt to make something
#   that is intelligent enough to be called
#           a potentate of the rose.
#
# This was adaped from tensorflow docs 
#   example code.
#
# And steffen owes me a deer if I can do this.
#
###############################################
import numpy as np
import tensorflow as tf

# Get randomized batch from the dataset
def make_batch(ins, outs, batchsize = 100):
    #ins = np.array(ins)
    #outs = np.array(outs)
    inbatch, outbatch = [], []
    for i in range(batchsize):
        idx = np.random.randint(1,ins.shape[0])
        inbatch.append(ins[idx])
        outbatch.append(outs[idx])
    outbatch = np.resize(outbatch, (batchsize, 1))
    return inbatch, outbatch

# Read dataset (shut up, it works :p)
def get_data():
    ins, outs = [], []
    dataset_file = open("rosepetals.dat", 'r')
    for line in dataset_file:
        petal_data=dataset_file.readline()
        tmp = []
        for s in petal_data.split():
            try:
                try:
                    if s[1] == ']':
                        tmp.append(int(s[0]))
                except IndexError:
                    pass
                if s[0] == '[':
                    tmp.append(int(s[1]))  
                elif s.isdigit():
                   tmp.append(int(s))
                else:
                    outs.append(float(s))
            except ValueError:
                pass
        ins.append(tmp)
    dataset_file.close()
    return np.array(ins), np.array(outs)

inputs, answers = get_data()

# Start TensorFlow part
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 5])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.zeros([5,1]))
b = tf.Variable(tf.zeros([1]))

sess.run(tf.initialize_all_variables())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
    batchx, batchy = make_batch(inputs, answers)
    train_step.run(feed_dict={x: batchx, y_:batchy})

# Evaluation of the model:

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batchx, batchy = make_batch(inputs, answers)

print("Accuracy: ", sess.run(accuracy, feed_dict={x: batchx, y_: batchy}))