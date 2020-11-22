import pickle

review_record = pickle.load(open('Electronics.dat', 'rb'))

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
pickle.dump(schema, open('Electronics.schema', 'wb'))