import tensorflow as tf
import numpy as np

'''
	data: the tensor containing the metadata of the user-item pairs
	batch_size: size of batch
	num_repeat: number of repeat of data
'''

def create_tf_dataset(data, batch_size, num_repeat):
	ds = tf.data.Dataset.from_tensor_slices(data)	
	ds = ds.shuffle()
	ds = ds.batch(batch_size)
	ds = ds.repeat(num_repeat)
	return ds

class NCF:	

	def __init__(self, schema, hp):	
		self.hp = hp
		self.schema = schema
		self.embedding = {}
		for k, v in schema['categorical'].items():
			self.embedding[k] = tf.get_variable(
				dtype = tf.float32,
				shape = (v['num_category'], v['num_units']),
				initializer = tf.contrib.layers.xavier_initializer()
			)
		
	def mlp(self, x, training=True):

		feature_list = []
		for k, v in self.schema['categorical'].items():
			emb = tf.nn.embedding_lookup(self.embedding[k], x[k])
			emb = tf.reshape(emb, [-1, v['num_units']])
		for k in self.schema['continuous']:
			feature_list.append(tf.reshape(x[k], [-1, 1]))
		feature = tf.concat(feature_list, axis = 1)

		layer_output = []
		for i in range(0, self.hp['num_layers']):
			output_units = self.hp['output_units'][i]
			if i = 0:
				input_v = feature
			else 
				input_v = layer_output[i - 1]
			layer_output[i] = tf.layers.dense(
				inputs = input_v,
				units = output_units,
				activation = tf.nn.relu,
				kernel_regularizer = tf.nn.l2_loss
			)
			prob = 1.0
			if training == True:
				prob = self.hp['keep_prob']
			layer_output[i] = tf.nn.dropout(layer_output[i], keep_prob = prob)
		
		out = tf.nn.softmax(layer_output[-1])
		return out