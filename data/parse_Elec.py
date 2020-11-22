import numpy as np
import json
import pickle
from tqdm import tqdm
import tensorflow as tf

item_cat = ['asin', 'main_cat', 'brand', 'date']
item_con = ['price']
cat_idx = {}
cat_cnt = {}

item_attr = {}

def parse_amazon_meta():
	filename = 'json/meta_Electronics.json'
	print('loading', filename)
	data = []
	f = open(filename)
	lines = f.readlines()

	print('collecting statistics...')
	for k in item_cat:
		cat_idx[k] = {}
		cat_cnt[k] = 0
	for l in tqdm(lines):
		record = json.loads(l)
		
		# if there is any missing value
		missing = False 
		for k in item_cat:
			if k not in record or record[k] == '':
				missing = True 
		for k in item_con:
			if k not in record or record[k] == '':
				missing = True
		if missing == True:
			continue		

		data.append(record)	
		
		for k, v in record.items():
			if k not in item_cat:
				continue
			if v not in cat_idx[k].keys():
				cat_idx[k][v] = cat_cnt[k]
				cat_cnt[k] += 1
	for k, v in cat_cnt.items():
		print(k, v) 

	print('total item number', len(lines))
	print('valid item number', len(data))

	print('genenrating metadata...')
	for record in tqdm(data):
		if record['asin'] == '':
			continue
		
		asin = record['asin']
		item_attr[asin] = {}	
		for k in item_cat:
			item_attr[asin][k] = cat_idx[k][record[k]]
		
		if record['price'] != '' and record['price'][0] == '$':
			record['price'] = record['price'][1:]
		item_attr[asin]['price'] = record['price']
	print('metadata generated')


def parse_amazon_review():
	filename = 'json/Electronics.json'
	print('loading', filename)
	data = []
	f = open(filename)
	lines = f.readlines()

	print('collecting statistics...')
	reviewer_id = {}
	reviewer_idx = 0
	for l in tqdm(lines):
		record = json.loads(l)

		if 'reviewerID' not in record or record['reviewerID'] == '':
			continue
		if record['asin'] not in cat_idx['asin']:
			continue

		data.append(record)
		rid = record['reviewerID']
		if rid not in reviewer_id:
			reviewer_id[rid] = reviewer_idx
			reviewer_idx += 1

	print('total review number', len(lines))
	print('valid review number', len(data))		

	print('generating user-item data')
	uidata = {}
	for k in item_cat:
		uidata[k] = []
	uidata['rid'] = []
	uidata['price'] = []
	uidata['rate'] = []
	for record in tqdm(data):
		rid = record['reviewerID']
		asin = record['asin']
		overall = record['overall']
		try:
			rate = float(overall)
		except:
			continue	
		uidata['rate'].append(rate)	

		if asin not in item_attr:
			continue
		for k in item_cat:
			uidata[k].append(item_attr[asin][k])

		uidata['rid'].append(reviewer_id[rid])

		try:
			price = float(item_attr[asin]['price'])
		except:
			price = 0.0 
		uidata['price'].append(price)

	outfile = open('Electronics.dat', 'wb')
	pickle.dump(uidata, outfile)
	print('user-item data generated')

parse_amazon_meta()
parse_amazon_review()
