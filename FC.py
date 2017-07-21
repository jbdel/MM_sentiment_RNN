import sys
import tensorflow as tf
from utils import *
from tensorflow.python.ops import variable_scope
from cells import *
import numpy as np
import sklearn.metrics


dataset = "train/MOSI/"


with open(dataset + 'target_filtered.txt') as f:
    data_y = f.read().splitlines()
    data_y = list(map(float, data_y))

randomize = np.load(dataset + "shuffle.npy")
data_y = np.array(data_y)
data_y = data_y[randomize]



encoder_inputs = tf.placeholder(tf.float32, [None, 1024*3])
y = tf.placeholder(tf.float32, [None, 2])
dropout_inputs_fc = tf.placeholder(tf.float32)

data_size = len(data_y)
test_size = int(len(data_y)/10)
train_size = data_size-test_size
print("train_size", train_size)
print("test_size", test_size)

k_fold = 10
batch_size = 100

n_input = 1024*3
n_hidden1 = 1024*2
n_hidden2 = 1024
n_hidden3 = 512
n_hidden4 = 256
n_hidden5 = 64
n_output = 2



w = {
    'h1': variable_scope.get_variable("h1", [n_input, n_hidden1]),
    'h2': variable_scope.get_variable("h2", [n_hidden1, n_hidden2]),
    'h3': variable_scope.get_variable("h3", [n_hidden2, n_hidden3]),
    'h4': variable_scope.get_variable("h4", [n_hidden3, n_hidden4]),
    'h5': variable_scope.get_variable("h5", [n_hidden4, n_hidden5]),
    'out': variable_scope.get_variable("out_h", [n_hidden5, n_output])
}
b = {
    'b1': variable_scope.get_variable("b1", [n_hidden1]),
    'b2': variable_scope.get_variable("b2", [n_hidden2]),
    'b3': variable_scope.get_variable("b3", [n_hidden3]),
    'b4': variable_scope.get_variable("b4", [n_hidden4]),
    'b5': variable_scope.get_variable("b5", [n_hidden5]),
    'out': variable_scope.get_variable("out_b", [n_output])
}


def multilayer_perceptron(x_, weights, biases, dropout):
    # Hidden layer with RELU activation
    for i in range(1, 6):
        hidden = tf.add(tf.matmul(x_, weights["h" + str(i)]), biases["b" + str(i)])
        x_ = tf.nn.relu(hidden)
        x_ = tf.nn.dropout(x_, dropout)
    out_layer = tf.matmul(x_, weights['out']) + biases['out']
    return out_layer


logits = multilayer_perceptron(encoder_inputs, w, b, dropout_inputs_fc)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

pred_y = tf.argmax(logits,1)
gold_y = tf.argmax(y,1)

# Evaluate model
correct_pred = tf.equal(pred_y, gold_y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



device = '/gpu:{}'.format(0)
tf_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.device(device):
    with tf.Session(config=tf_config) as sess:
        print('using device: {}'.format(device))


        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(v.name)

        test_accuracies = []
        f1_scores = []

        for i in range(k_fold):
            # Launch the graph
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())

            text_file = np.load('train/MOSI/repr/sentences_repr_{}_kfold{}.npy'.format('text',i))
            video_file = np.load('train/MOSI/repr/sentences_repr_{}_kfold{}.npy'.format('video',i))
            audio_file = np.load('train/MOSI/repr/sentences_repr_{}_kfold{}.npy'.format('audio',i))

            data_x = np.concatenate([text_file, audio_file, video_file], 1)
            data_x = np.array(data_x)
            data_x = data_x[randomize]



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

                x_batch = []
                y_batch = []
                for _ in range(batch_size):
                    r = random.randrange(0, len(train_x), 2)
                    x_batch.append(train_x[r])
                    y_batch.append(train_y[r])
                y_batch = vector_to_one_hot(y_batch)
                input_feed = {}
                input_feed[encoder_inputs] = x_batch
                input_feed[y] = y_batch
                input_feed[dropout_inputs_fc] = 0.5

                sess.run(optimizer, input_feed)
                loss = sess.run(cost, input_feed)

                #eval on dev dataset
                dropout_ = 1.0
                x_batch = []
                y_batch = []

                for idx in range(len(dev_x)):
                    x_batch.append(dev_x[idx])
                    y_batch.append(dev_y[idx])

                y_batch = vector_to_one_hot(y_batch)

                input_feed = {}
                input_feed[encoder_inputs] = x_batch
                input_feed[y] = y_batch
                input_feed[dropout_inputs_fc] = 1.0


                acc, pred_y_, gold_y_ = sess.run([accuracy, pred_y, gold_y], input_feed)
                accuracies.append(acc)

                if acc > best_acc:
                    best_acc = acc

                if loss < best_loss and (best_loss - loss) > 1e-6:
                    best_loss = loss
                    stop_counter = 0
                if (stop_counter == 100 or loss < 1e-4) and loss < 0.2:
                    break
                stop_counter += 1



                print("Iter " + str(step * batch_size) + " Epoch : " + str(int((step * batch_size)/len(data_x)))+", Training minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Dev Accuracy= " + \
                      "{:.5f}".format(acc))

                step += 1

            assert np.amax(np.array(accuracies)) == best_acc, "what ?"
            print("Optimization Finished! Best dev accuracies: {:.5f}".format(best_acc))
            print("Now running this best model to test set")
            # dropout_ = 1.0
            x_batch = []
            y_batch = []
            for idx in range(len(test_x)):
                x_batch.append(test_x[idx])
                y_batch.append(test_y[idx])

            y_batch = vector_to_one_hot(y_batch)

            input_feed = {}
            input_feed[encoder_inputs] = x_batch
            input_feed[y] = y_batch
            input_feed[dropout_inputs_fc] = 1.0

            acc, pred_y_, gold_y_ = sess.run([accuracy, pred_y, gold_y], input_feed)
            f1_score = sklearn.metrics.f1_score(gold_y_, pred_y_)
            f1_scores.append(f1_score)
            test_accuracies.append(acc)

            print("K-fold", i + 1, "Test accuracy = {:.5f}".format(acc))
            print("K-fold", i + 1, "F1 score = {:.5f}".format(f1_score))

        print("Average accuracies : ", np.mean(test_accuracies))
        print("Average f1_scores : ", np.mean(f1_scores))


