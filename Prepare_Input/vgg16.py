import tensorflow as tf
import tensorlayer as tl
import os
import numpy as np
import h5py

class vgg16(object):

	def conv_layers(self, image):
		'''
		with tf.name_scope('preprocess'):
			mean = tf.constant([123.68,116.779,103.939],dtype=tf.float32, shape=[1,1,1,3], name='img_mean')
			image.outputs = image.outputs - mean
		'''
		# conv1
		network = tl.layers.Conv2dLayer(image, 
										act=tf.nn.relu, 
										shape=[3,3,3,64],
										strides=[1,1,1,1],
										padding='SAME',
										name='conv1_1')
		network = tl.layers.Conv2dLayer(network,
										act=tf.nn.relu,
										shape=[3,3,64,64],
										strides=[1,1,1,1],
										padding='SAME',
										name='conv1_2')
		network = tl.layers.PoolLayer(network,
										ksize=[1,2,2,1],
										strides=[1,2,2,1],
										padding='SAME',
										pool=tf.nn.max_pool,
										name='pool1')
		
		#conv2
		network = tl.layers.Conv2dLayer(network, 
										act=tf.nn.relu, 
										shape=[3,3,64,128],
										strides=[1,1,1,1],
										padding='SAME',
										name='conv2_1')
		network = tl.layers.Conv2dLayer(network, 
										act=tf.nn.relu, 
										shape=[3,3,128,128],
										strides=[1,1,1,1],
										padding='SAME',
										name='conv2_2')
		network = tl.layers.PoolLayer(network,
										ksize=[1,2,2,1],
										strides=[1,2,2,1],
										padding='SAME',
										pool=tf.nn.max_pool,
										name='pool2')
		#conv3
		network = tl.layers.Conv2dLayer(network, 
										act=tf.nn.relu, 
										shape=[3,3,128,256],
										strides=[1,1,1,1],
										padding='SAME',
										name='conv3_1')
		network = tl.layers.Conv2dLayer(network, 
										act=tf.nn.relu, 
										shape=[3,3,256,256],
										strides=[1,1,1,1],
										padding='SAME',
										name='conv3_2')
		network = tl.layers.Conv2dLayer(network, 
										act=tf.nn.relu, 
										shape=[3,3,256,256],
										strides=[1,1,1,1],
										padding='SAME',
										name='conv3_3')
		network = tl.layers.PoolLayer(network,
										ksize=[1,2,2,1],
										strides=[1,2,2,1],
										padding='SAME',
										pool=tf.nn.max_pool,
										name='pool3')
		# conv4
		network = tl.layers.Conv2dLayer(network, 
										act=tf.nn.relu, 
										shape=[3,3,256,512],
										strides=[1,1,1,1],
										padding='SAME',
										name='conv4_1')
		network = tl.layers.Conv2dLayer(network, 
										act=tf.nn.relu, 
										shape=[3,3,512,512],
										strides=[1,1,1,1],
										padding='SAME',
										name='conv4_2')
		network = tl.layers.Conv2dLayer(network, 
										act=tf.nn.relu, 
										shape=[3,3,512,512],
										strides=[1,1,1,1],
										padding='SAME',
										name='conv4_3')
		network = tl.layers.PoolLayer(network,
										ksize=[1,2,2,1],
										strides=[1,2,2,1],
										padding='SAME',
										pool=tf.nn.max_pool,
										name='pool4')
		# conv5
		network = tl.layers.Conv2dLayer(network, 
										act=tf.nn.relu, 
										shape=[3,3,512,512],
										strides=[1,1,1,1],
										padding='SAME',
										name='conv5_1')
		network = tl.layers.Conv2dLayer(network, 
										act=tf.nn.relu, 
										shape=[3,3,512,512],
										strides=[1,1,1,1],
										padding='SAME',
										name='conv5_2')
		network = tl.layers.Conv2dLayer(network, 
										act=tf.nn.relu, 
										shape=[3,3,512,512],
										strides=[1,1,1,1],
										padding='SAME',
										name='conv5_3')
		network = tl.layers.PoolLayer(network,
										ksize=[1,2,2,1],
										strides=[1,2,2,1],
										padding='SAME',
										pool=tf.nn.max_pool,
										name='pool5')
		pool5 = network.outputs
		return network, pool5

	def fc_layers(self, net):
		network = tl.layers.FlattenLayer(net, name='flatten')
		network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc1_relu')
		network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc2_relu')
		last2 = network.outputs
		network = tl.layers.DenseLayer(network, n_units=1000, act=tf.nn.relu, name='fc3_relu')
		return network, last2
	def __init__(self, images, sess, spatial=False):
		x = tf.placeholder(tf.float32, [None, 224,224,3])
		image_in = tl.layers.InputLayer(x, name='input')
		network_cnn, pool5 = self.conv_layers(image_in)
		network, last2 = self.fc_layers(network_cnn)
		
		tl.layers.initialize_global_variables(sess)
		
		network.print_params(False)
		#network.print_layers(False)
		try:
			assert os.path.isfile('data/vgg16_weight/vgg16_weights.zip')
			weights = np.load('data/vgg16_weight/vgg16_weights.zip')
		except AssertionError as e:
			raise Exception('Please provide pretrained weight').with_traceback(e.__traceback__)
		
		params = []
		for val in sorted(weights.items()):
			params.append(val[1])
		
		tl.files.assign_params(sess, params, network)
		if spatial:
			self.conv = sess.run(pool5, feed_dict={x:images})
		else:
			
			self.features = sess.run(last2, feed_dict={x:images})
		'''
		batch = 10
		epoch = int(images.shape[0]/batch)
		features = []
		for i in range(epoch+1):
			if i == epoch:
				image_batch = images[i*batch:]
			else:
				image_batch = images[batch:(i+1)*batch]
			self.features = sess.run(last2, feed_dict={x:image_batch})
			features.append(features)
		np.save('test.npy', features)
		'''
		
if __name__ == "__main__":
	filename = 'data\qa_data.h5'
	sess = tf.Session()
	with h5py.File(filename, 'r') as hp:
		keys = [i for i in hp.keys()]
		images = hp['image_data'][()]
	images = images.transpose(0,2,3,1)
	vgg = vgg16(images, sess)
