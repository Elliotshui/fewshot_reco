import pandas as pd
import numpy as np
import json
import pickle
from tqdm import tqdm
import tensorflow as tf

def parse_amazon_meta():
	filename = 'json/meta_Electronics.json'
	print('loading', filename)
	data = []
	f = open(filename)
	lines = f.readlines()

	print('collecting statistics...')
	fields = {}
	idx = {}
	key = ['asin', 'main_cat', 'brand', 'date']
	for k in key:
		fields[k] = {}
		idx[k] = 0
	for l in tqdm(lines):
		record = json.loads(l)
		data.append(record)
		for k, v in record.items():
			if k not in key:
				continue
			if v == '':
				continue
			if v not in fields[k].keys():
				fields[k][v] = idx[k]
				idx[k] += 1
	for k, v in idx.items():
		print(k, v) 

	print('item number', len(data))

	print('genenrating metadata...')
	metadata = {
		'data': [],
		'asin_id': {}
	} 
	for record in tqdm(data):
		if record['asin'] == '':
			continue
		category_val = {}
		for k in key:
			if k not in record or record[k] == '':
				category_val[k] = idx[k]
			else:
				category_val[k] = fields[k][record[k]]
		if 'price' in record:
			if record['price'] == '':
				category_val['price'] = -1.0
			else:
				if record['price'][0] == '$':
					record['price'] = record['price'][1:]
				category_val['price'] = record['price']
		metadata['data'].append(category_val)
	metadata['asin_id'] = fields['asin']

	outfile = open('meta_Electronics.pickle', 'wb')
	pickle.dump(metadata, outfile)
	print('metadata pickle generated')


def parse_amazon_review():
	filename = 'json/Electronics.json'
	print('loading', filename)
	data = []
	f = open(filename)
	lines = f.readlines()

	print('collecting statistics...')
	reviewer_id = {}
	idx = 0
	for l in tqdm(lines):
		record = json.loads(l)
		data.append(record)
		if 'reviewerID' not in record or record['reviewerID'] == '':
			continue
		rid = record['reviewerID']
		if rid not in reviewer_id:
			reviewer_id[rid] = idx
			idx += 1
	print('review number', len(data))		

	print('generating metadata')
	metadata = {
		'data': [],
	}
	for record in tqdm(data):
		if 'reviewerID' not in record or record['reviewerID'] == '':
			continue
		val = {}
		val['rid'] = reviewer_id[record['reviewerID']]
		if 'asin' in record:
			val['asin'] = record['asin']
		if 'overall' in record:
			val['overall'] = record['overall']
		metadata['data'].append(val)
	outfile = open('Electronics.pickle', 'wb')
	pickle.dump(metadata, outfile)
	print('review pickle generated')
	print(metadata['data'][0])

#parse_amamzon_meta()
parse_amazon_review()
