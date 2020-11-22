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
	'keep_prob': 0.8,
	'lambda' : 0.01,
	'lr': 1e-4
}

model = NCF(schema, hp)

BATCH_SIZE = 64
NUM_REPEAT = 1
x = create_tf_dataset(review_record, schema, BATCH_SIZE, NUM_REPEAT)
print('dataset created')

iterator = x.make_one_shot_iterator()
one_element = iterator.get_next()

feature, out = model.mlp(one_element, True)
loss, update = model.train(one_element, out)
print(tf.trainable_variables())

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	layer_loss = model.train_guided(one_element, out)
	l = sess.run([layer_loss])
	print(l)