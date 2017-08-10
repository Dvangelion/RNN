import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import rnn_cell,rnn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt

# Learning Parameters
learning_rate = 0.001
training_iters = 3
test_iters = 151
train_data_size = 4000
test_iters = 10
display_step = 5
batch_size = 1
logs_path = '/tmp/tensorflow_logs/example'

# Network Parameters
num_hidden = 200 # hidden layer num of features
num_skills = 378 # maximum number of skill id
input_size = 1


# Read dataframe
rnn_df = pd.read_csv('skill_builder_data_corrected.csv',low_memory=False,header=0)
rnn_df = rnn_df[pd.notnull(rnn_df['skill_name'])]


skill_name = rnn_df['skill_name'].dropna()
skill_id = rnn_df['skill_id'].dropna()
user_id = rnn_df['user_id'].dropna()
correct = rnn_df['correct'].dropna()

#new dataframe contains skill_name, skill_id.. etc.
dataframe = pd.concat([skill_name,skill_id,user_id,correct], axis=1)

#list of user_id
id_list = dataframe['user_id'].unique().tolist()
id_list.sort()


skill_id_list = []
correct_list = []
seq_len = []


for id in id_list:
    sub_df = dataframe.loc[dataframe['user_id'] == id]
    skill_id_list.append(sub_df['skill_id'].tolist())
    correct_list.append(sub_df['correct'].tolist())
    seq_len.append(len(sub_df['skill_id'].tolist()))


def split_list(x, correct):
    unique_values = list(set(x))
    index = []
    for i in unique_values:
        index.append(x.index(i))
    index.sort()

    result = []
    correct_list = []
    for i in range(len(index) - 1):
        result += [x[index[i]:index[i+1]]]
        correct_list += [correct[index[i]:index[i+1]]]
    result += [x[index[-1]:]]
    correct_list += [correct[index[-1]:]]


    return result, correct_list

def split_nested_list(nest_list, nest_correct_list):
    result = []
    correct_list = []
    seq_len = []
    for i,j in zip(nest_list, nest_correct_list):
        result += [split_list(i,j)[0]]
        correct_list += [split_list(i,j)[1]]


    for i in result:
        length = []
        for j in i:
            length += [len(j)]
        seq_len += [length]

    return result, correct_list, seq_len

def padding_zeros(nest_list, seq_len):
    """
    :param nest_list: a nest list of skill id of a single student
    :param seq_len:
    :return:
    """
    max_length = max(seq_len)
    for i in nest_list:
        for j in range(len(i),max_length):
            i.append(0)

    return np.array(nest_list).astype(np.float32)

skill_id, correct, sequence_len = split_nested_list(skill_id_list, correct_list)

prob_len = [sum(i) for i in sequence_len]
max_problem_nums = max(prob_len)
num_steps = max_problem_nums

# Placeholder variables
x = tf.placeholder("float",[num_steps])
y = tf.placeholder("float",[None])
seqlen = tf.placeholder(tf.int32, [None])
target_id = tf.placeholder(tf.int32, [None])


weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_skills]))
}

biases = {
    'out': tf.Variable(tf.random_normal([num_skills]))
}

hidden1 = rnn_cell.LSTMCell(num_hidden, input_size)
hidden1 = rnn_cell.DropoutWrapper(hidden1, output_keep_prob = .6)
cell = rnn_cell.MultiRNNCell([hidden1])

input = tf.to_int32(x)
input = tf.one_hot(input, 378)
input = tf.reshape(x,[batch_size, num_steps ,input_size])
input = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, input)]


lstm_cell = rnn_cell.LSTMCell(num_hidden, forget_bias= 0.5)
GRU_cell = rnn_cell.GRUCell(num_hidden)

outputs, states = rnn.rnn(cell, input, dtype = tf.float32)
output = tf.reshape(tf.concat(1, outputs), [-1, num_hidden])


logits = tf.matmul(output, weights['out']) + biases['out']
logits = tf.reshape(logits,[-1])
selected_logits = tf.gather(logits, target_id)

pred = tf.sigmoid(selected_logits)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.scalar_summary("loss", cost)
#tf.scalar_summary("accuracy", accuracy)
merged_summary_op = tf.merge_all_summaries()

# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    index = 0
    pred_labels = []
    actual_labels = []

    # Keep training until reach max iterations
    while index + batch_size  < training_iters:

        batch_x, batch_y, batch_seqlen = skill_id_list[index % train_data_size], \
                                correct_list[index % train_data_size], sequence_len[index % train_data_size]

        batch_y = batch_y[:-1]

        batch_x = np.zeros(num_steps)
        target_list = []

        for i in range(len(skill_id_list[index]) - 1):
            if(batch_y[i] == 0):
                label_index = skill_id_list[index][i]
            else:
                label_index = skill_id_list[index][i] + num_skills
            batch_x[i] = label_index

            target_list.append(i * num_skills + int(skill_id_list[index][i + 1]))



        if index % display_step == 0 and len(batch_y) > 1:

            # Calculate batch loss
            loss = sess.run(cost, feed_dict = {x: batch_x, y: batch_y, seqlen: batch_seqlen, target_id:target_list})
            summary = sess.run(merged_summary_op, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen, target_id:target_list})

            pred_labels = sess.run(pred, feed_dict = {x: batch_x, y: batch_y, seqlen: batch_seqlen, target_id:target_list})

            rmse = sqrt(mean_squared_error(batch_y, pred_labels))
            fpr, tpr, thresholds = metrics.roc_curve(batch_y, pred_labels, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            r2 = r2_score(batch_y, pred_labels)

            print "Iter " + str(index) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss), 'rmse: ',rmse, 'auc: ',auc, 'r2: ',r2

            summary_writer.add_summary(summary, index)

            index += batch_size


        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen, target_id:target_list})

        index += batch_size

    print "Optimization Finished. "


    # #Use last 151 user as test data
    # test_data = skill_id_list[-151:]
    # test_label = correct_list[-151:]
    # test_seq_len = sequence_len[-151:]
    #
    # step = 0
    #
    # while step < test_iters:
    #     batch_x, batch_y, batch_seqlen = test_data[step], test_label[step], test_seq_len[step]
    #     batch_x = np.zeros(num_steps)
    #     target_list = []
    #
    #     for i in range(len(test_data[step]) - 1):
    #         if(batch_y[i] == 0):
    #             label_index = skill_id_list[step][i]
    #         else:
    #             label_index = skill_id_list[step][i] + num_skills
    #         batch_x[i] = label_index
    #
    #         target_list.append(i * num_skills + int(skill_id_list[index][i + 1]))
    #
    #
    # print "Accuracy: ", sess.run(tf.reduce_mean(acc_list))
    #
    # print "Run the command line:\n" \
    #       "--> tensorboard --logdir=/tmp/tensorflow_logs " \
    #       "\nThen open http://0.0.0.0:6006/ into your web browser"


