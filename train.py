import tensorflow as tf
import numpy as np
import pickle

from model import NCF, create_tf_dataset

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

review_record = pickle.load(open('data/Electronics.dat', 'rb'))

for k, v in schema['categorical'].items():	
	v['num_category'] = 0
	for val in review_record[k]:
		v['num_category'] = max(v['num_category'], val)
	v['num_category'] += 1
	v['num_units'] = 16

print(schema)

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

out = model.mlp(one_element, True)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	score = sess.run(out)
	print(score)