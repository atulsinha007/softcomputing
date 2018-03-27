import tensorflow as tf
import numpy as np
import load_data

trainx, trainy, testx, testy = load_data.load_data()

n_nodes_hl1 = 2

n_classes = 2
batch_size = 2

x = tf.placeholder('float',[None,2])
y = tf.placeholder('float')

def neural_net_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([2,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_classes])),
		'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']),hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']
	
	return output

def train_neural_network(x):

	prediction = neural_net_model(x)
	print(prediction)
	# cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

	# optimizer = tf.train.AdamOptimizer().minimize(cost)

	# hm_epochs = 10

	# with tf.Session() as sess:
	# 	sess.run(tf.initialize_all_variables())

	# 	# for epoch in range(hm_epochs):
	# 	# 	epoch_loss = 0
	# 		# for _ in range(int((len(trainx))/batch_size)):
	# 		# 	epoch_x,epoch_y  = trainx.next_batch(batch_size)
	# 		# 	_,epoch_c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
	# 		# 	epoch_loss += epoch_c
	# 		# print('Epoch', epoch, 'completed out of ', hm_epochs, 'loss: ', epoch_loss)
	# 	for epoch in range(hm_epochs):
	# 	   _, c = sess.run([optimizer, cost], feed_dict = {x:trainx, y:trainy})

	# 	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
	# 	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	# 	print('Accuracy:', accuracy.eval({x:testx, y:testy}))

train_neural_network(x)