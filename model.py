import argparse
import json
import string
import tensorflow as tf
import h5py
import numpy as np
import pickle
import math

one_file = 'data/final_input.pickle'
with open(one_file, 'rb') as f:
    data = pickle.load(f)
data.keys()
data['train'].keys()

resume = True
if resume:
    embed_file = 'Checkpoint/embed.npy'
    other_param = 'Checkpoint/baseline_without_dropout.npz'
    embed_load = np.load(embed_file)
    baseline_without_dropout = np.load(other_param)
# read train data
train_q = data['train']['question']
train_mc = data['train']['multiple']
train_l = data['train']['label']
train_im = data['train']['im_id']
# read val data
val_q = data['val']['question']
val_mc = data['val']['multiple']
val_l = data['val']['label']
val_im = data['val']['im_id']
# read test data
test_q = data['test']['question']
test_mc = data['test']['multiple']
test_l = data['test']['label']
test_im = data['test']['im_id']
# concatenate train and val data
train_q = np.concatenate([train_q, val_q], axis=0)[:5000]
train_mc = np.concatenate([train_mc, val_mc], axis=0)[:5000]
train_l = np.concatenate([train_l, val_l],axis=0)[:5000]
train_im = np.concatenate([train_im, val_im], axis=0)[:5000]
# read feature
features = data['feature']
features = features.reshape(-1,49,512)

assert len(train_q) == len(train_mc) == len(train_l) == len(train_im)
train_l = np.argmax(train_l, axis=1)
test_l = np.argmax(test_l, axis=1)
del data

vocab_size = 3007
embed_size = 300
batch_size = 32
epoch = 100

tf.reset_default_graph()
tl.layers.clear_layers_name()
sess = tf.InteractiveSession()

mc = tf.placeholder(tf.int32, [None, 4,5], name='mc')
question = tf.placeholder(tf.int32, [None, 15], name='question')
image_id = tf.placeholder(tf.int32, [None,], name='image_id')
y_ = tf.placeholder(tf.int32, [None, ], name='label')
im_embed = tf.placeholder(tf.float32, [28653, 49,512], name='image_embed')

# the correct answer is at the last row.
#'''
with tf.name_scope('embedding'):
    im_feature = tf.nn.embedding_lookup(im_embed, image_id)
    word_embed = tf.Variable(tf.truncated_normal([vocab_size, embed_size],
                                                stddev=1.0/math.sqrt(embed_size)),tf.float32,name='word_embed')
    mc_vec = tf.nn.embedding_lookup(word_embed, mc)
    mc_vec = tf.reduce_sum(mc_vec, axis=2)
    mc_vec = tf.reshape(mc_vec, [-1,1200])

    question_vec = tf.nn.embedding_lookup(word_embed, question)

# Attention Mechanism
Wc = tf.Variable(tf.truncated_normal([300, 49],stddev=1.0/math.sqrt(49)), tf.float32)
QWc = tf.einsum('ijk,kl->ijl', question_vec, Wc)
C = tf.einsum('ijk,ikl->ijl', QWc, im_feature)

k=60
Wv = tf.Variable(tf.truncated_normal([k, 49],stddev=1.0/math.sqrt(49)), tf.float32)
Wq = tf.Variable(tf.truncated_normal([k,300],stddev=1.0/math.sqrt(300)), tf.float32)
WvV = tf.einsum('kl,ilj->ikj', Wv, im_feature)
WqQ = tf.einsum('kl,ilj->ikj', Wq, tf.transpose(question_vec,[0,2,1]))
Hv = tf.tanh(WvV + tf.einsum('ijk,ikl->ijl', WqQ, C))
Hq = tf.tanh(WqQ + tf.einsum('ijk,ikl->ijl', WvV, tf.transpose(C,[0,2,1])))

whv = tf.Variable(tf.truncated_normal([1,k],stddev=1.0/math.sqrt(k)), tf.float32)
whq = tf.Variable(tf.truncated_normal([1,k],stddev=1.0/math.sqrt(k)), tf.float32)
av = tf.nn.softmax(tf.einsum('kl,ilj->ikj', whv, Hv), axis=1)
aq = tf.nn.softmax(tf.einsum('kl,ilj->ikj', whq, Hq), axis=1)

v_ = tf.reduce_sum(av*im_feature, axis=2)
q_ = tf.reduce_sum(aq*tf.transpose(question_vec,[0,2,1]), axis=2)


q_im_mc = tf.concat([q_,v_,mc_vec], axis=1)
'''
q_im_mc1 = tf.concat([question_, im_feature, mc1_], axis=1)
q_im_mc2 = tf.concat([question_, im_feature, mc2_], axis=1)
q_im_mc3 = tf.concat([question_, im_feature, mc3_], axis=1)
q_im_mc4 = tf.concat([question_, im_feature, mc4_], axis=1)
q_im_mc = tf.concat([q_im_mc1,q_im_mc2,q_im_mc3,q_im_mc4], axis=1)
#'''


sess.run(tf.global_variables_initializer())     

network = tl.layers.InputLayer(q_im_mc, name='input')
network = tl.layers.DenseLayer(network, n_units=8192, act=tf.nn.relu, name='hidden1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout1')
#network = tl.layers.DenseLayer(network, n_units=8192, act=tf.identity, name='hidden2')
network = tl.layers.DenseLayer(network, n_units=4, act=tf.identity, name='output')


logits = network.outputs
cost = tl.cost.cross_entropy(logits, y_, name='cost')
# add regularization
L2 = 0
for p in tl.layers.get_variables_with_name('/W',True,True):
    L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
cost += L2

correct_pred = tf.equal(tf.argmax(logits,axis=1,output_type=tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.variable_scope('learning_rate'):
    lr = tf.Variable(0.0004, trainable=False)

train_op = tf.train.AdamOptimizer(lr,
                                  beta1=0.8,
                                  beta2=0.999,
                                  epsilon=1e-08).minimize(cost)

tl.layers.initialize_global_variables(sess)

resume=True
if resume:
    sess.run(tf.assign(word_embed, embed_load))

network.print_params(False)
network.print_layers()



def minibatches(q, mc, im, l, batch_size):
    for start_idx in range(0, len(q)-batch_size+1, batch_size):
        idx = slice(start_idx, start_idx + batch_size)
        yield q[idx], mc[idx], im[idx], l[idx]
for i in range(1, epoch+1):
    if i % 5 == 0 and i!=1:
        current_lr = sess.run(lr)
        sess.run(tf.assign(lr, current_lr*0.9))

    for train_q_a, train_mc_a, train_im_a, train_l_a in minibatches(train_q, train_mc, train_im, train_l, batch_size):
        feed_dict={mc:train_mc_a,
                   question:train_q_a,
                   image_id:train_im_a,
                   y_:train_l_a,
                   im_embed:features}
        feed_dict.update(network.all_drop)
        sess.run(train_op, feed_dict=feed_dict)



    test_loss, test_acc, batch_test = 0,0,0
    for test_q_a, test_mc_a, test_im_a, test_l_a in minibatches(train_q, train_mc, train_im, train_l, batch_size):
        feed_dict={mc:test_mc_a,
                   question:test_q_a,
                   image_id:test_im_a,
                   y_:test_l_a,
                   im_embed:features}
        dp_dict = tl.utils.dict_to_one(network.all_drop)
        feed_dict.update(dp_dict)
        loss_batch, acc_batch = sess.run([cost, accuracy], feed_dict=feed_dict)
        test_loss += loss_batch
        test_acc += acc_batch
        batch_test += 1
    print('Epoch: %i' % i +'/'+str(epoch))
    print('test_loss: %f' % (test_loss/batch_test))
    print('test_acc: %f' % (test_acc/batch_test))

    if i % 1 == 0 and i != 0:
        print('Saving data' + '!'*10)
        tl.files.save_npz_dict(network.all_params, name='model.npz', sess=sess)
