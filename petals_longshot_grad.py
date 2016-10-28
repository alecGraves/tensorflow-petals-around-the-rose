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
    ins = np.array(ins)
    outs = np.array(outs)
    # Convert dataset answers to better format for pur purposes
    for i in range(len(outs)):
        outs[i] = 1/(1+np.exp(-1*outs[i]/20.0))
        if(i % 10000 == 0):
            print(ins[i], outs[i])
    return ins, outs


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(-0.1, shape=shape)
    return tf.Variable(initial)


#get training data:
inputs, answers = get_data()


# Start TensorFlow part
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 5])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

W = weight_variable((5,1))
b = bias_variable([1])

y = tf.sigmoid(tf.matmul(x, W) + b)
#y = tf.tanh(tf.matmul(x, W) + b)

#experimental changes:
cross_entropy = tf.reduce_mean(tf.abs(tf.sub((y), (y_))))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#cross_entropy = -tf.reduce_sum(y_*tf.log(y), reduction_indices=[1])
#best working one:
#cross_entropy = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(7e-4).minimize(cross_entropy)

sess.run(tf.initialize_all_variables())

# Evaluation of the model:

correct_prediction = tf.greater(0.05, tf.abs(tf.sub((y), (y_))))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#accuracy = tf.cast(correct_prediction, tf.float32)

batchx, batchy = make_batch(inputs, answers, 1000)

current_accuracy = sess.run(accuracy, feed_dict={x: batchx, y_: batchy})

fittest_model = [sess.run(W), sess.run(b), current_accuracy]

while (current_accuracy < 1.0):
    batchx, batchy = make_batch(inputs, answers, 10000)
    train_step.run(feed_dict={x: batchx, y_:batchy})
    current_accuracy = sess.run(accuracy, feed_dict={x: batchx, y_: batchy})
    if (fittest_model[2] < current_accuracy):
        fittest_model = [sess.run(W), sess.run(b), current_accuracy]
        print(fittest_model)