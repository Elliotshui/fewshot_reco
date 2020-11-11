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

pickle.dump(schema, open('data/Electronics.schema', 'wb'))
'''

hp = {
	'num_layers': 1,
	'output_units': [1],
	'keep_prob': 0.8
}

model = NCF(schema, hp)

BATCH_SIZE = 64
NUM_REPEAT = 1
x = create_tf_dataset(review_record, schema, BATCH_SIZE, NUM_REPEAT)
print('dataset created')

iterator = x.make_one_shot_iterator()
one_element = iterator.get_next()

feature, out = model.mlp(one_element, True)
variable_names = [v.name for v in tf.trainable_variables()]
print(variable_names)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	f, score = sess.run([feature, out])
	print(f)