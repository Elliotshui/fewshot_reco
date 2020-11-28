import tensorflow as tf
import numpy as np
import pickle

from model import NCF, create_tf_dataset

review_record = pickle.load(open('data/Electronics.dat', 'rb'))
schema = pickle.load(open('data/Electronics.schema', 'rb'))

'''
schema = {
	'categorical': {
		'rid': {},
		'asin' : {},
		'main_cat': {},
		'brand': {},
		'date': {}
	},
	'continuous': [
		'price'
	],
	'target': 'rate'
}


for k, v in schema['categorical'].items():	
	v['num_category'] = 0
	for val in review_record[k]:
		v['num_category'] = max(v['num_category'], val)
	v['num_category'] += 1
	v['num_units'] = 16

print(schema)
pickle.dump(schema, open('data/Electronics.schema', 'wb'))
'''

hp = {
	'num_layers': 1,
	'output_units': [1],
	'keep_prob': 0.8,		# keep probability for dropout layer
	'reg_lambda' : 0.01,	# lambda for regularization term
	'l_lambda': 0.1,		# lambda for layer loss
	'lr': 1e-3,				# learning rate for normal training
	'lr_tune': 1e-4			# learning rate for finetuning 
}

tb_dir = './tensorboard/'

model = NCF(schema, hp)

BATCH_SIZE = 64
NUM_REPEAT = 1
x = create_tf_dataset(review_record, schema, BATCH_SIZE, NUM_REPEAT)
print('dataset created')

iterator = x.make_one_shot_iterator()
one_element = iterator.get_next()

out = model.mlp(one_element, True)
out1 = model.mlp(one_element, True)

loss, update = model.train(one_element, out)
loss_tune, update_tune = model.tune(one_element, out1)

print(tf.trainable_variables())

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	lt = sess.run(loss_tune)
	print(lt)

	'''
	tb_writer = tf.summary.FileWriter(tb_dir + 'model')
	tb_writer.add_graph(sess.graph)
	'''