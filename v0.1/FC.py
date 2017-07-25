import sys
import tensorflow as tf
from utils import *
from tensorflow.python.ops import variable_scope
from cells2 import *
import pickle
import numpy as np

dataset = "train/MOSI/"
text_file = np.load("train/MOSI/sentences_repr_text.npy")
# video_file = np.load("train/MOSI/sentences_repr_video.npy")
audio_file = np.load("train/MOSI/audio.npy")

data_x = np.concatenate([text_file,audio_file],1)


# data_x = text_file

with open(dataset+'list.txt', 'r') as f:
    list = f.readlines()

with open(dataset+'list_filtered.txt', 'r') as f:
    list_filtered = f.readlines()

#delete according to index
with open(dataset+'target.txt', 'r') as f:
    data_y_ = f.readlines()
    data_y = []
    for i,y in enumerate(data_y_):
        if(list[i] in list_filtered):
            data_y.append(float(y.strip()))

# assert len(text_file) == len(video_file), "NOK"
assert len(data_x) == len(data_y), "NOK2"

data_x = np.array(data_x)
data_y = np.array(data_y)


randomize = np.arange(len(data_x))
np.random.shuffle(randomize)
data_x = data_x[randomize]
data_y = data_y[randomize]


encoder_inputs = tf.placeholder(tf.float32, [None, 1024*2])

y = tf.placeholder(tf.float32, [None, 2])

data_size = len(data_x)
test_size = int(len(data_x)/10)
train_size = data_size-test_size
print("train_size", train_size)
print("test_size", test_size)

k_fold = 10
batch_size = 100

n_input = 1024*2
n_hidden = 1024*2
n_hidden2 = 512
n_hidden3 = 256
n_hidden4 = 64
n_output = 2



w = {
    # 'h1': variable_scope.get_variable("h1", [n_input, n_hidden]),
    'h2': variable_scope.get_variable("h2", [n_hidden, n_hidden2]),
    'h3': variable_scope.get_variable("h3", [n_hidden2, n_hidden3]),
    'h4': variable_scope.get_variable("h4", [n_hidden3, n_hidden4]),
    'out': variable_scope.get_variable("out_h", [n_hidden4, n_output])
}
b = {
    # 'b1': variable_scope.get_variable("b1", [n_hidden]),
    'b2': variable_scope.get_variable("b2", [n_hidden2]),
    'b3': variable_scope.get_variable("b3", [n_hidden3]),
    'b4': variable_scope.get_variable("b4", [n_hidden4]),
    'out': variable_scope.get_variable("out_b", [n_output])
}


def multilayer_perceptron(x_, weights, biases, dropout):
    # Hidden layer with RELU activation
    for i in range(2, 5):
        hidden = tf.add(tf.matmul(x_, weights["h" + str(i)]), biases["b" + str(i)])
        x_ = tf.nn.relu(hidden)
        x_ = tf.nn.dropout(x_, dropout)
    out_layer = tf.matmul(x_, weights['out']) + biases['out']
    return out_layer

# input_conc = tf.concat([encoder_inputs_text,encoder_inputs_video],1)

pred = multilayer_perceptron(encoder_inputs, w, b, 1.0)



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)


# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



device = '/gpu:{}'.format(2)
tf_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.device(device):
    with tf.Session(config=tf_config) as sess:
        print('using device: {}'.format(device))


        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(v.name)

        test_accuracies = []
        max_test_accuacy=-1.0

        for i in range(k_fold):
            # Launch the graph
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())


            # si on est pas au dernier k_fold, on prend commme taille de test test_size
            if i < k_fold - 1:
                start = i * test_size
                end = (i + 1) * test_size

                test_x = data_x[start:end]
                test_y = data_y[start:end]
                # get index to remove from data_x for train
                r = np.arange(start, end)
                train_x = np.delete(data_x, r, 0)
                train_y = np.delete(data_y, r, 0)
                #train must now be dived into train and dev (last 10% == test_size)
                dev_x = train_x[len(train_x)-test_size: len(train_x)]
                dev_y = train_y[len(train_y)-test_size: len(train_y)]
                r = np.arange(len(train_x)-test_size, len(train_x))
                train_x = np.delete(train_x, r, 0)
                train_y = np.delete(train_y, r, 0)



            if i == k_fold - 1:
                start = i * test_size
                end = len(data_x)
                test_x = data_x[start:end]
                test_y = data_y[start:end]
                # get index to remove test from data_x for train
                r = np.arange(start, end)
                train_x = np.delete(data_x, r, 0)
                train_y = np.delete(data_y, r, 0)
                # train must now be dived into train and dev (last 10% == test_size)
                dev_x = train_x[len(train_x) - test_size: len(train_x)]
                dev_y = train_y[len(train_y) - test_size: len(train_y)]
                r = np.arange(len(train_x) - test_size, len(train_x))
                train_x = np.delete(train_x, r, 0)
                train_y = np.delete(train_y, r, 0)

            print("Starting k-fold number", i+1)
            print("K-fold", i+1, "train size =", len(train_x))
            print("K-fold", i+1, "dev size =", len(dev_x))
            print("K-fold", i+1, "test size =", len(test_x))


            accuracies = []
            best_loss = sys.maxsize
            stop_counter = 0
            step = 0
            best_acc = 0.0
            while True:

                # dropout_rnn_ = 1.0-dropout_rate_rnn

                x_batch = []
                y_batch = []
                for _ in range(batch_size):
                    r = random.randrange(0, len(train_x), 2)
                    x_batch.append(train_x[r])
                    y_batch.append(train_y[r])
                y_batch = vector_to_one_hot_real(y_batch)
                input_feed = {}
                input_feed[encoder_inputs] = x_batch
                input_feed[y] = y_batch
                # input_feed[dropout_inputs_fc] = dropout_fc_

                sess.run(optimizer, input_feed)
                pred_ = sess.run(pred, input_feed)

                loss = sess.run(cost, input_feed)

                # rand_ = random.randint(0,10)
                # if rand_ == 5:
                #eval on dev dataset
                dropout_ = 1.0
                x_batch = []
                y_batch = []
                for _ in range(batch_size):
                    r = random.randrange(0, len(dev_x), 2)
                    x_batch.append(dev_x[r])
                    y_batch.append(dev_y[r])

                y_batch = vector_to_one_hot_real(y_batch)

                input_feed = {}
                input_feed[encoder_inputs] = x_batch
                input_feed[y] = y_batch

                acc = sess.run(accuracy, input_feed)
                # prediction = sess.run(pred, input_feed)

                #
                accuracies.append(acc)
                if acc > best_acc:
                    best_acc = acc
                    saver.save(sess, 'models/kk', global_step=0)

                if loss < best_loss and (best_loss-loss)> 1e-6:
                    best_loss = loss
                    stop_counter = 0
                if loss > 0.2:
                    stop_counter = 0
                if stop_counter == 100 or loss < 1e-4:
                    break
                stop_counter +=1



                print("Iter " + str(step * batch_size) + " Epoch : " + str(int((step * batch_size)/len(data_x)))+", Training minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Dev Accuracy= " + \
                      "{:.5f}".format(acc))

                step += 1

            assert np.amax(np.array(accuracies)) == best_acc, "what ?"
            print("Optimization Finished! Best dev accuracies: {:.5f}".format(best_acc))
            print("Now running this best model to test set")
            saver.restore(sess, "models/kk-0")
            dropout_ = 1.0
            x_batch = []
            y_batch = []
            for idx in range(batch_size):
                r = random.randrange(0, len(test_x), 2)
                x_batch.append(test_x[r])
                y_batch.append(test_y[r])

            y_batch = vector_to_one_hot_real(y_batch)

            input_feed = {}
            input_feed[encoder_inputs] = x_batch
            input_feed[y] = y_batch
            acc = sess.run(accuracy, input_feed)

            test_accuracies.append(acc)
            print("K-fold", i + 1, "Test accuracy = {:.5f}".format(acc))
            if acc + best_acc > max_test_accuacy:
                max_test_accuacy = acc + best_acc
                saver.save(sess, 'models/best-test0')

        for i in test_accuracies:
            print(i)
        print("Average accuracies : ", np.mean(test_accuracies))



