import h5py
import numpy as np
import pickle
import argparse

def shuffle_data(in_file, feature_path, out_file):
	dataset= h5py.File(in_file, 'r')
	ans = dataset['answer_label'].value
	mc = dataset['mc_label'].value
	qa_end_ix = dataset['qa_end_ix'].value
	qa_start_ix = dataset['qa_start_ix'].value
	ques = dataset['question_label'].value
	split = dataset['split'].value
	im_id = dataset['image_id'].value
	im_fl = [[i]*(qa_end_ix[i]-qa_start_ix[i]+1) for i in range(im_id.shape[0])]
	im_fl = np.array(sum(im_fl,[]))
	split_fl = [[split[i]]*(qa_end_ix[i]-qa_start_ix[i]+1) for i in range(im_id.shape[0])]
	split = np.array(sum(split_fl, []))

	N = ans.shape[0]

	multiple = np.concatenate([np.expand_dims(ans, axis=1), mc], axis=1)
	label = np.zeros([N,4])
	label[range(N), [0]] = 1

	batch = int(np.ceil(N/4))
	correct_pos = [0,]
	for i in range(1,4):
		if i == 3:
			data_slice = slice(i*batch,N)
		else:
			data_slice = slice(i*batch,(i+1)*batch)
		while True:
			shuffle = np.random.permutation(np.arange(4)).tolist()
			idx = shuffle.index(0)
			if idx not in correct_pos:
				correct_pos.append(idx)
				break
		mul_temp = multiple[data_slice].transpose(1,2,0)
		multiple[data_slice] = mul_temp[shuffle].transpose(2,0,1)
		label[data_slice,:] = label[data_slice,shuffle]

	shuffle = np.random.permutation(np.arange(N))
	multiple = multiple[shuffle]
	label = label[shuffle]
	ques = ques[shuffle]
	split = split[shuffle]
	im_id_fl = im_fl[shuffle]

	dict_f = {}
	for i in range(3):
		if i == 0:
			cate = 'train'
			print('Writing %s data.............' % cate)
		elif i==1:
			cate = 'val'
			print('Writing %s data.............' % cate)
		elif i==2:
			cate = 'test'
			print('Writing %s data.............' % cate)
		multiple_temp = multiple[split==i]
		label_temp = label[split==i]
		ques_temp = ques[split==i]
		im_id_fl_temp = im_id_fl[split==i]
		dict_ = {'multiple':multiple_temp, 'label':label_temp,'question':ques_temp,'im_id':im_id_fl_temp}
		dict_f[cate] = dict_

	print('Loading Image Feature.........')
	feature = np.load(feature_path)
	dict_f['feature'] = feature

	print('Dumping Data Into File............')
	with open(out_file,'wb') as hh:
		pickle.dump(dict_f, hh, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--path',default='data/qa_data.h5', help='path to raw dataset h5 file')
	parser.add_argument('--output_pickle', default='data/final_input.pickle', help='output pickle file')
	parser.add_argument('--spatial_feature', default='data/conv_feature.npy',help='path to spatial feature')
	args = parser.parse_args()
	params = vars(args)
	
	shuffle_data(params['path'],params['spatial_feature'], params['output_pickle'])
	
	
