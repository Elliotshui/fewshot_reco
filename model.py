import tensorflow as tf
import numpy as np

'''
	data: the tensor containing the metadata of the user-item pairs
	batch_size: size of batch
	num_repeat: number of repeat of data
'''

def create_tf_dataset(data, schema, batch_size, num_repeat):
	def gen():
		for k, v in data.items():
			num_sample = len(v)	
			break
		for i in range(0, num_sample):
			ls = {}
			for k, v in data.items():
				if k == 'rate':
					if v[i] >= 4.0:
						ls[k] = 1.0
					else:
						ls[k] = 0.0
				else:
					ls[k] = v[i]
			yield ls

	output_types = {}
	output_shapes = {}
	for k in schema['categorical']: 
		output_types[k] = tf.int32
	for k in schema['continuous']:
		output_types[k] = tf.float32
	output_types[schema['target']] = tf.float32

	ds = tf.data.Dataset.from_generator(
			gen,
			output_types = output_types
		)	
	ds = ds.shuffle(buffer_size = 1024)
	ds = ds.batch(batch_size)
	ds = ds.repeat(num_repeat)
	return ds

def get_var_by_name(name):
	for v in tf.global_variables():
		if v.name == name:
			return v

class NCF:	

	def __init__(self, schema, hp):	
		self.hp = hp
		self.schema = schema
		self.embedding = {}
		for k, v in schema['categorical'].items():
			self.embedding[k] = tf.Variable(
				tf.zeros(shape = [v['num_category'], v['num_units']]),
				name = k + '_embedding'
			)
		
	def mlp(self, x, training=True):

		feature_list = []
		for k, v in self.schema['categorical'].items():
			emb = tf.nn.embedding_lookup(self.embedding[k], x[k])
			emb = tf.reshape(emb, [-1, v['num_units']])
			feature_list.append(emb)
		for k in self.schema['continuous']:
			feature_list.append(tf.reshape(x[k], [-1, 1]))
		feature = tf.concat(feature_list, axis = 1)

		layer_output = []
		for i in range(0, self.hp['num_layers']):
			output_units = self.hp['output_units'][i]
			if i == 0:
				input_v = feature
			else:
				input_v = layer_output[i - 1]
			lout = (tf.layers.dense(
				inputs = input_v,
				units = output_units,
				activation = tf.nn.relu,
				name = 'dense_' + str(i),
			))
			prob = 1.0
			if training == True:
				prob = self.hp['keep_prob']
			layer_output.append(tf.nn.dropout(lout, keep_prob = prob))
		
		out = tf.nn.sigmoid(layer_output[-1])
		out = tf.reshape(out, [-1])
		return feature, out
	
	def train(self, x, out):

		loss_error = tf.reduce_mean(tf.square(x[self.schema['target']] - out), axis = 0)
		reg_collection = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
		loss_reg = tf.add_n(reg_collection)
		loss = loss_error + self.hp['lambda'] * loss_reg

		optimizer = tf.train.AdamOptimizer(
			learning_rate = self.hp['lr']
		)
		update = optimizer.minimize(loss)
		
		return loss, update

	def train_guided(self, x, out):

		layer_ll = []

		for i in range(0, self.hp['num_layers']):
			kernel = get_var_by_name('dense_' + str(i) + '/kernel:0')
			bias = get_var_by_name('dense_' + str(i) + '/bias:0')
			kernel_o = tf.Variable(tf.zeros(1), trainable = False)
			bias_o = tf.Variable(tf.zeros(1), trainable = False)
			kernel_o = tf.assign(kernel_o, kernel, validate_shape = False)
			bias_o = tf.assign(bias_o, bias, validate_shape = False)
			layer_ll.append(tf.reduce_sum(tf.square(kernel_o - kernel)))
			layer_ll.append(tf.reduce_sum(tf.square(bias_o - bias)))

		layer_loss = tf.add_n(layer_ll)
		
		return layer_loss 