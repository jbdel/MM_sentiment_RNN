import sys
import tensorflow as tf
from utils import *
from tensorflow.python.ops import variable_scope
from cells2 import *
import copy
import sklearn.metrics
import argparse


#making ids file

dataset = "train/MOSI/"


with open(dataset+"vocab.en", "r") as f:
    l = f.readlines()
    voc = {}
    for i,r in enumerate(l):
        voc[i] = r.strip("\n")

#word -> id
inv_voc = {v: k for k, v in voc.items()}
with open(dataset+"train.ids.en", "w") as f:
    with open(dataset+"train.en", "r") as f2:
        l = f2.readlines()
        for r in l:
            new_line = [str(inv_voc[w]) for w in r.split()]
            f.write(' '.join(new_line))
            f.write("\n")





# data
print ("Reading training data")
data_x_fixed, data_y_fixed, src_max = read_data(dataset+"train.ids.en", dataset+"target.txt")

data_x = copy.deepcopy(data_x_fixed)
data_y = copy.deepcopy(data_y_fixed)

randomize = np.arange(len(data_x))
np.random.shuffle(randomize)
data_x = data_x[randomize]
data_y = data_y[randomize]

PAD_ID = 0

data_size = len(data_x)
test_size = int(len(data_x)/10)
train_size = data_size-test_size
print("data_size", data_size)
print("train_size", train_size)
print("test_size", test_size)
k_fold = 10


# model settings
dtype = tf.float32
initializer = tf.truncated_normal_initializer(stddev=0.0001)
cell_fw = GRUCell(512)
cell_bw = GRUCell(512)



dropout_rate_rnn = 0.2
dropout_rate_fc = 0.0

dropout_rnn = tf.Variable(1 - dropout_rate_rnn, trainable=False, name='dropout_keep_prob')
dropout_rnn_input = tf.Variable(1.0, trainable=False, name='input_dropout_keep_prob')

embedding_size = 630
embedding_classes = len(inv_voc)

embedding = vs.get_variable(
    "embedding", [embedding_classes, embedding_size],
    initializer=initializer,
    dtype=dtype)

dropout_output = dropout_rnn
dropout_state = dropout_rnn
dropout_input = dropout_rnn_input

encoder_cell_fw = EmbeddingDropoutWrapper(cell_fw, embedding, output_keep_prob=dropout_output, state_keep_prob=dropout_state,
                                    input_keep_prob=dropout_input, variational_recurrent=False, input_size=embedding_size, dtype=dtype)
encoder_cell_bw = EmbeddingDropoutWrapper(cell_bw, embedding, output_keep_prob=dropout_output, state_keep_prob=dropout_state,
                                    input_keep_prob=dropout_input, variational_recurrent=False, input_size=embedding_size, dtype=dtype)

batch_size = 100
max_step = src_max



# encoder_inputs =[]
# for i in range(max_step):
#     encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
#                                                     name="encoder{0}".format(i)))

encoder_inputs = tf.placeholder(tf.int32, [None, max_step, 1])
y = tf.placeholder(tf.float32, [None, 2])
x_lengths = tf.placeholder(tf.int32, [None])
dropout_inputs_rnn = tf.placeholder(tf.float32)
dropout_inputs_fc = tf.placeholder(tf.float32)


n_input = 2048
n_hidden = 1024
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


def RNN(x, x_lengths, weights, biases, dropout_rnn, dropout_fc):

        x = tf.unstack(x, max_step, 1)

        encoder_cell_fw.set_dropout(dropout_rnn)
        encoder_cell_bw.set_dropout(dropout_rnn)

        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(encoder_cell_fw, encoder_cell_bw, x,
                                                     dtype=tf.float32)
        output_gru = outputs[-1]

        #
        def multilayer_perceptron(x_, weights, biases, dropout):
            # Hidden layer with RELU activation
            for i in range(2,5):
                hidden = tf.add(tf.matmul(x_, weights["h"+str(i)]), biases["b"+str(i)])
                x_ = tf.nn.relu(hidden)
                x_ = tf.nn.dropout(x_, dropout)
            out_layer = tf.matmul(x_, weights['out']) + biases['out']
            return out_layer

        out_fc = multilayer_perceptron(output_gru,w,b, dropout_fc)

        return out_fc, output_gru

logits, output = RNN(encoder_inputs, x_lengths, w, b, dropout_inputs_rnn, dropout_inputs_fc)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)


# Evaluate model
pred_y = tf.argmax(logits,1)
gold_y = tf.argmax(y,1)
correct_pred = tf.equal(pred_y, gold_y)
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


            # getting right data for kfold
            if i < k_fold - 1:
                start = i * test_size
                end = (i + 1) * test_size


            if i == k_fold - 1:
                start = i * test_size
                end = len(data_x)

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

                dropout_rnn_ = 1.0-dropout_rate_rnn
                dropout_fc_ = 1.0-dropout_rate_fc
                x_batch, y_batch, x_batch_lengths = get_batch_random(train_x, train_y, batch_size, max_step, PAD_ID)
                x_batch = x_batch.reshape((batch_size, max_step, 1))
                y_batch = vector_to_one_hot_real(y_batch)

                input_feed = {}
                input_feed[encoder_inputs] = x_batch
                input_feed[y] = y_batch
                input_feed[x_lengths] = x_batch_lengths
                input_feed[dropout_inputs_rnn] = dropout_rnn_
                input_feed[dropout_inputs_fc] = dropout_fc_

                sess.run(optimizer, input_feed)

                loss = sess.run(cost, input_feed)

                #eval on dev dataset
                dropout_ = 1.0
                x_batch, y_batch, x_batch_lengths = get_batch_fixed(dev_x, dev_y, max_step, PAD_ID)
                test_size_ = len(dev_x)
                x_batch = x_batch.reshape((test_size_, max_step, 1))
                y_batch = vector_to_one_hot_real(y_batch)

                input_feed = {}
                input_feed[encoder_inputs] = x_batch
                input_feed[y] = y_batch
                input_feed[x_lengths] = x_batch_lengths
                input_feed[dropout_inputs_rnn] = dropout_
                input_feed[dropout_inputs_fc] = dropout_

                acc, pred_y_, gold_y_ = sess.run([accuracy, pred_y, gold_y], input_feed)


                accuracies.append(acc)
                if acc > best_acc:
                    best_acc = acc
                    saver.save(sess, 'models/save-txt', global_step=0)

                if loss < best_loss and (best_loss-loss)> 1e-6:
                    best_loss = loss
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
            x_batch, y_batch, x_batch_lengths = get_batch_fixed(test_x, test_y, max_step, PAD_ID)
            test_size = len(test_x)
            x_batch = x_batch.reshape((test_size, max_step, 1))
            y_batch = vector_to_one_hot_real(y_batch)
            input_feed = {}
            input_feed[encoder_inputs] = x_batch
            input_feed[y] = y_batch
            input_feed[x_lengths] = x_batch_lengths
            input_feed[dropout_inputs_rnn] = dropout_
            input_feed[dropout_inputs_fc] = dropout_
            acc = sess.run(accuracy, input_feed)

            test_accuracies.append(acc)
            print("K-fold", i+1, "Test accuracy = {:.5f}".format(acc))
            if acc+best_acc > max_test_accuacy:
                saver.save(sess,'models/best-test0')
        for i in test_accuracies:
            print(i)
        print("Average accuracies : ", np.mean(test_accuracies))
        print('Best test-dev :', max_test_accuacy)
        print("Now extracting features for every sentence")
        saver.restore(sess, "models/best-test0")
        dropout_ = 1.0
        x_batch, y_batch, x_batch_lengths = get_batch_fixed(data_x_fixed, data_y_fixed, max_step, PAD_ID)
        test_size = len(x_batch)
        x_batch = x_batch.reshape((test_size, max_step, 1))
        y_batch = vector_to_one_hot_real(y_batch)
        input_feed = {}
        input_feed[encoder_inputs] = x_batch
        input_feed[y] = y_batch
        input_feed[x_lengths] = x_batch_lengths
        input_feed[dropout_inputs_rnn] = dropout_
        input_feed[dropout_inputs_fc] = dropout_
        acc,outs = sess.run([accuracy, output], input_feed)
        print("Overall           "
              " = {:.5f}".format(acc))
        print(outs.shape)
        np.save('train/MOSI/sentences_repr_text.npy', outs)

