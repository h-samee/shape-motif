#!/usr/bin/python
import numpy as np
import os
import sys
import math
import glob
from scipy import stats
############ global consts ###########
max_rank = 100

#read_max_count = (1000 * 999)/2

beta = 1.0/3

min_sigma_count = 0.1
max_sigma_count = 2
sigma_count_increment = 0.1

motif_overlap_thr = 0.75
total_set_overlap_thr = 0.75

min_neglog_p_val_thr = 10
######################################
feature_name = ""

def does_contain(motif_a, motif_b, len_a, len_b):
	# returns 1 if motif_a contains motif_b
	motif_a_total_sites = 0
	motif_b_total_sites = 0

	for i in motif_a:
		if i not in motif_b:
			motif_a_total_sites += len(motif_a[i])

	for i in motif_b:
		if i not in motif_a:
			motif_b_total_sites += len(motif_b[i])

	motif_a_overlapping_sites_with_b = 0
	motif_b_overlapping_sites_with_a = 0

	for i in motif_a:
		if i in motif_b:
			motif_a_total_sites += len(motif_a[i])
			motif_b_total_sites += len(motif_b[i])

			lst_a = motif_a[i]
			lst_b = motif_b[i]
			
			for start_a, strand_a in lst_a:
				for start_b, strand_b in lst_b:
					end_a = start_a + len_a - 1
					end_b = start_b + len_b - 1	
					if start_b > end_a or start_a > end_b:
						continue
					else:
						if start_b >= start_a:
							overlap_amount = min(end_a, end_b) - start_b + 1
						else:
							overlap_amount = min(end_a, end_b) - start_a + 1

						if not float(overlap_amount)/len_a < motif_overlap_thr:
							motif_a_overlapping_sites_with_b += 1
						if not float(overlap_amount)/len_b < motif_overlap_thr:
							motif_b_overlapping_sites_with_a += 1

	#if not float(motif_a_overlapping_sites_with_b)/motif_a_total_sites < total_set_overlap_thr:
	#	return 2
	if not float(motif_b_overlapping_sites_with_a)/motif_b_total_sites < total_set_overlap_thr:
		return 1
	return 0

def remove_redundant_motifs(all_motifs):
	'''
	# format of record of individual motifs:
	(test_neglog_p_val, test_f_score, best_neglog_p_val, best_f_score, \
	 motif_id, motif_len, best_motif_as_range, best_k, \
	 best_tpr, best_fpr, test_tpr, test_fpr, \
	 best_chip_occurrences, test_chip_occurrences))
	'''
	non_redundant_motifs = []
	n_motifs = len(all_motifs)
	motif_redundancy_status = [False] * n_motifs

	for i in range(n_motifs):
		if motif_redundancy_status[i] == True:
			print('possible error in logic: i-th motif\'s redundacny status is supposed to be False')
			continue
		cur_motif = all_motifs[i]
		for j in range(i + 1, n_motifs):
			if motif_redundancy_status[j] == True:
				print('possible error in logic: j-th motif\'s redundancy status is supposed to be False')
				continue
	
			next_motif = all_motifs[j]

			# check if cur_motif should be marked as redundant
			cur_motif_len = abs(cur_motif['motif_len'])
			next_motif_len = abs(next_motif['motif_len'])
			
			# do not compare for redundancy if the lengths are not comparable
			if (int)(cur_motif_len/5) != (int)(next_motif_len/5):
				continue
			# check if next_motif contains cur_motif
			# note: by order of sorting, next_motif has smaller fpr than cur_motif
			redundancy_result = does_contain( \
			next_motif['test_chip_occurrences'], cur_motif['test_chip_occurrences'], \
			next_motif_len, cur_motif_len)
			if redundancy_result == 1:
				motif_redundancy_status[i] = True
				print("motif %d (in the list sorted according to FPRs) found to be redundant due to motif %d" % (i, j))
				break
				
	for i in range(n_motifs):
		if motif_redundancy_status[i] == False:
			non_redundant_motifs.append(all_motifs[i])
	return non_redundant_motifs

def does_motif_match_window(motif_as_range, window):
	window_len = len(window)
	assert window_len == len(motif_as_range)
	
	#check forward direction
	mismatch = False
	for i in range(window_len):
		data_val = window[i]
		min_val = motif_as_range[i][0]
		max_val = motif_as_range[i][1]
		if data_val < min_val or data_val > max_val:
			mismatch = True
			break
	
	if mismatch == True:
	#if mismatch in forward direction, check reverse direction
		window_ = window[::-1]
		for i in range(window_len):
			data_val = window_[i]
			min_val = motif_as_range[i][0]
			max_val = motif_as_range[i][1]
			if data_val < min_val or data_val > max_val:
				return False, '*'
		return True, '-'
	else:
		return True, '+'

def list_motif_occurrences(cur_shape_data, motif_as_range):
	seq_len = len(cur_shape_data)
	window_size = len(motif_as_range)
	occurrence_list = []
	for i in range(seq_len - window_size + 1):
		cur_window = cur_shape_data[i : i + window_size]
		match_status, strand = does_motif_match_window(motif_as_range, cur_window)
		if match_status == True:
			occurrence_list.append((i, strand))
	
	return occurrence_list

def compute_f_score(n_chip_seq_with_shape, n_chip_seq, n_ctrl_seq_with_shape, n_ctrl_seq):
	'''
	   |   P'   |   N'   |
	---|--------|--------|
	 P | a = TP | b = FN |
	---|--------|--------|
	 N | c = FP | d = TN |
	---|--------|--------|
	'''

	if not (n_chip_seq_with_shape > 0):
		return 0
	
	a = n_chip_seq_with_shape
	b = n_chip_seq - n_chip_seq_with_shape
	c = n_ctrl_seq_with_shape
	d = n_ctrl_seq - n_ctrl_seq_with_shape
	precision = float(a)/(a+c)
	recall = float(a)/(a+b)
	f_score = (1 + beta * beta) * precision * recall/(beta * beta * precision + recall)
	return f_score

def compute_tpr_fpr(n_chip_seq_with_shape, n_chip_seq, n_ctrl_seq_with_shape, n_ctrl_seq):
	'''
	   |   P'   |   N'   |
	---|--------|--------|
	 P | a = TP | b = FN |
	---|--------|--------|
	 N | c = FP | d = TN |
	---|--------|--------|
	'''

	if not (n_chip_seq_with_shape > 0):
		return (0, 0)
	a = n_chip_seq_with_shape
	b = n_chip_seq - n_chip_seq_with_shape
	c = n_ctrl_seq_with_shape
	d = n_ctrl_seq - n_ctrl_seq_with_shape
	tpr = float(a)/(a + b)
	fpr = float(c)/(c + d)
	return (tpr, fpr)

def count_occurrences(motif_as_range, shape_data):
	"""
	Note: 
	-- This function returns:
	---- number of sequences that contain the motif and
	---- locations of the motif's occurrences (overlapping occurrences allowed) in each sequence (as a dictionary)
	"""
	n_seq = len(shape_data)
	#seq_len = len(shape_data[0])
	window_size = len(motif_as_range)
	
	occurrence_count = 0
	occurrence_dict = {}

	for i in range(n_seq):
		cur_shape_data = shape_data[i]
		lst = list_motif_occurrences(cur_shape_data, motif_as_range)
		if len(lst) > 0:
			occurrence_count += 1
		occurrence_dict[i] = lst

	return occurrence_count, occurrence_dict

def compute_ranges(instances, sigma_count):
	n_seq = len(instances)
	window_size = len(instances[0])
	ranges = []
	
	for i in range(window_size):
		vals_at_i = []
		for j in range(n_seq):
			data_val = instances[j][i]
			vals_at_i.append(data_val)
		
		
		mean_at_i = np.mean(np.array(vals_at_i))
		std_at_i = np.std(np.array(vals_at_i))

		min_val = mean_at_i - sigma_count * std_at_i 
		max_val = mean_at_i + sigma_count * std_at_i 
		
		ranges.append((min_val, max_val))
	
	return ranges

def read_instances(instances_file_name):
	'''
	returns a list of tuples of the form:
	(motif_id, motif_len, [instance_1, instance_2, ...])
	where each instance is a list of floats
	'''
	all_instances = []
	cur_instances = []
	motif_id = -1
	motif_len = -1
	
	with open(instances_file_name) as f:
		for line in f:
			line = line.strip()
			if line[0] == '#':
				if len(cur_instances) > 0:
					all_instances.append((motif_id, motif_len, cur_instances))
				cur_instances = []
				[motif_id, motif_len] = map(int, line[1:].split(','))
			else:
				vals = map(float, line.split())
				cur_instances.append(vals)

	if len(cur_instances) > 0:
		all_instances.append((motif_id, motif_len, cur_instances))

	return all_instances

def read_shape_data(file_name):
	bed_lines = []
	shape_data = []
	with open(file_name) as f:
		for line in f:
			vals = line.split()
			if len(vals) != 5:
				print('Possible error in the .dat file of shape profile: some fields missing in data file integrating .bw and bed file')
			else:
				chromosome = vals[0]
				start_coord = int(vals[1])
				end_coord = int(vals[2])

				data_len = int(vals[-2])
				if vals[-1].upper().find('NA') >= 0:
					continue
				data_vals = map(float, vals[-1].split(','))
				if len(data_vals) != data_len:
					print('Error: mismatch in bwtool generated data and data-len')
				else:
					shape_data.append(data_vals)
				bed_lines.append([chromosome, start_coord, end_coord])
	return shape_data, bed_lines

def main(argv = None):
	if argv is None:
		argv = sys.argv
	
	## input args
	chip_data_file_name = argv[1] 
	ctrl_data_file_name = argv[2] 
	test_chip_data_file_name = argv[3]
	test_ctrl_data_file_name = argv[4]
	output_prefix = argv[5]
	global feature_name
	feature_name = argv[6]
	instances_file_names = argv[7:]
	
	print('------------------  %s  ----------------------' % (output_prefix))
	## read shape data and bed file
	# chip peaks (train)
	chip_shape_data, chip_bed_lines = read_shape_data(chip_data_file_name)
	n_chip_seq = len(chip_shape_data)
	# control peaks (train)
	ctrl_shape_data, ctrl_bed_lines = read_shape_data(ctrl_data_file_name)
	n_ctrl_seq = len(ctrl_shape_data)
	# chip peaks (test)
	test_chip_shape_data, test_chip_bed_lines = read_shape_data(test_chip_data_file_name)
	n_test_chip_seq = len(test_chip_shape_data)
	# control peaks (test)
	test_ctrl_shape_data, test_ctrl_bed_lines = read_shape_data(test_ctrl_data_file_name)
	n_test_ctrl_seq = len(test_ctrl_shape_data)

	total_seq = n_chip_seq + n_ctrl_seq
	total_test_seq = n_test_chip_seq + n_test_ctrl_seq
	
	## set up the list of sigma-counts to optimize over
	sigma_counts = list(np.linspace(min_sigma_count, max_sigma_count, \
	math.ceil((max_sigma_count - min_sigma_count)/sigma_count_increment) + 1))
	
	## build a summary for each motif from its instances		
	all_motifs = []
	
	# iterate over each .instance file
	for instances_file_name in instances_file_names:
		# read instances of all motifs in the current .instance file
		all_instances = read_instances(instances_file_name)
		# all_instances is a list of 3-tuples: (motif_id, motif_len, cur_instances)
		print("read %d instances from %s" % (len(all_instances), instances_file_name))
		# iterate over each motif's instances
		prev_motif_len = 0
		for instances in all_instances:
			motif_id = instances[0]
			motif_len = instances[1]
			if motif_len == prev_motif_len:
				motif_len = -motif_len
			cur_instances = instances[2]

			# find the best value of k such that \mu + k\sigma yields the best F-score
			best_k = None
			best_f_score = -float("inf")
			best_tpr = None
			best_fpr = None
			best_neglog_p_val = None
			best_motif_as_range = None
			best_chip_occurrences = None
			
			for k in sigma_counts:
				motif_as_range = compute_ranges(cur_instances, k)
				n_chip_seq_with_shape, chip_occurrences = count_occurrences(motif_as_range, chip_shape_data)
				
				if n_chip_seq_with_shape > 0:
					n_ctrl_seq_with_shape, ctrl_occurrences = count_occurrences(motif_as_range, ctrl_shape_data)
					n_seq_with_shape = n_chip_seq_with_shape + n_ctrl_seq_with_shape
					
					f_score = compute_f_score(n_chip_seq_with_shape, n_chip_seq, n_ctrl_seq_with_shape, n_ctrl_seq)
					(tpr, fpr) = compute_tpr_fpr(n_chip_seq_with_shape, n_chip_seq, n_ctrl_seq_with_shape, n_ctrl_seq)
					
					sf_p_val = stats.hypergeom.sf(n_chip_seq_with_shape - 1, total_seq, n_chip_seq, n_seq_with_shape)
					neglog_p_val = -1
					if sf_p_val > 0:
						neglog_p_val = -math.log10(sf_p_val)
					else:
						neglog_p_val = 1000

					if f_score > best_f_score:
						best_f_score = f_score
						best_k = k
						best_tpr = tpr
						best_fpr = fpr
						best_neglog_p_val = neglog_p_val
						best_motif_as_range = motif_as_range
						best_chip_occurrences = chip_occurrences
			#if there is an optimized motif in the training set, 
			#then compute f-score & p-val on test (validation) set and append it
			if best_k is not None:
				print("After computing best_k: fscore = %f, tpr = %f, fpr = %f, neglog_p = %f" % (best_f_score, best_tpr, best_fpr, best_neglog_p_val))
				n_test_chip_seq_with_shape, test_chip_occurrences = count_occurrences(best_motif_as_range, test_chip_shape_data)
				n_test_ctrl_seq_with_shape, test_ctrl_occurrences = count_occurrences(best_motif_as_range, test_ctrl_shape_data)
				n_test_seq_with_shape = n_test_chip_seq_with_shape + n_test_ctrl_seq_with_shape
				
				#f-score
				test_f_score = compute_f_score(n_test_chip_seq_with_shape, n_test_chip_seq, n_test_ctrl_seq_with_shape, n_test_ctrl_seq)
				(test_tpr, test_fpr) = compute_tpr_fpr(n_test_chip_seq_with_shape, n_test_chip_seq, n_test_ctrl_seq_with_shape, n_test_ctrl_seq)
				#p-val
				sf_p_val = stats.hypergeom.sf(n_test_chip_seq_with_shape - 1, total_test_seq, n_test_chip_seq, n_test_seq_with_shape)
				test_neglog_p_val = -1
				if sf_p_val > 0:
					test_neglog_p_val = -math.log10(sf_p_val)
				else:
					test_neglog_p_val = 1000
				print("best_k:%f,test_neglog_p_val:%f,%d,%d,%d,%d,hypergeom_pars:%d,%d,%d,%d" % (best_k,test_neglog_p_val,n_test_chip_seq_with_shape, n_test_chip_seq, n_test_ctrl_seq_with_shape, n_test_ctrl_seq, n_test_chip_seq_with_shape - 1, total_test_seq, n_test_chip_seq, n_test_seq_with_shape))
				if not (test_neglog_p_val < min_neglog_p_val_thr):
					cur_entry = {}
					cur_entry['test_neglog_p_val'] = test_neglog_p_val
					cur_entry['test_f_score'] = test_f_score 
					cur_entry['best_neglog_p_val'] = best_neglog_p_val
					cur_entry['best_f_score'] = best_f_score 
					cur_entry['motif_id'] = motif_id 
					cur_entry['motif_len'] = motif_len
					cur_entry['best_motif_as_range'] = best_motif_as_range
					cur_entry['best_k'] = best_k
					cur_entry['best_tpr'] = best_tpr
					cur_entry['best_fpr'] = best_fpr
					cur_entry['test_tpr'] = test_tpr
					cur_entry['test_fpr'] = test_fpr
					cur_entry['best_chip_occurrences'] = best_chip_occurrences
					cur_entry['test_chip_occurrences'] = test_chip_occurrences
					all_motifs.append(cur_entry)
					'''
					all_motifs.append((test_neglog_p_val, test_f_score, best_neglog_p_val, best_f_score, \
						 motif_id, motif_len, best_motif_as_range, best_k, \
						 best_tpr, best_fpr, test_tpr, test_fpr, \
						 best_chip_occurrences, test_chip_occurrences))
					'''
					print("final records: pval(tr) = %f\tpval(t) = %f\tf(tr) = %f\tf(t) = %f\tlen = %d\tk = %f\ttpr(tr) = %f\tfpr(tr) = %f\ttpr(t) = %f\tfpr(t) = %f\n" \
					% (best_neglog_p_val, test_neglog_p_val,\
						best_f_score, test_f_score,\
						abs(motif_len), best_k,\
						best_tpr, best_fpr, test_tpr, test_fpr))
			prev_motif_len = motif_len
	###
	print('-------------- sorting the motifs in descending order of fpr (test) ------------------------')
	sorted_motifs = sorted(all_motifs, key = \
	lambda x: (-x['test_fpr'], x['test_tpr'], abs(x['motif_len'])))
	#sorted_motifs = sorted(all_motifs, key = lambda x: (-x[-3], x[-4], abs(x[5])))
	print("after sorting:")
	print("best fpr, tpr, and motif-lengths:")
	for motif in sorted_motifs:
		print("%f %f %d" % (motif['test_fpr'], motif['test_tpr'], abs(motif['motif_len'])))
		#print("%f %f %d" % (motif[-3], motif[-4], abs(motif[5])))
	

	### 
	print('-------------- removing redundant motifs ------------------------')
	non_redundant_motifs = remove_redundant_motifs(sorted_motifs)
	print("number of non-redundant motifs = %d" % (len(non_redundant_motifs)))
	print("best p-vals, fprs, tprs, f-scores, and motif-lengths:")
	for motif in non_redundant_motifs:
		print("%f %f %f %f %d" % \
		(motif['test_neglog_p_val'], \
		motif['test_fpr'], motif['test_tpr'], \
		motif['test_f_score'], abs(motif['motif_len'])))
	###
	print('--------------- sort non-redundant motifs -------------------')
	final_motifs = sorted(non_redundant_motifs, key = \
	lambda x: (x['test_fpr'], -x['test_tpr'], abs(x['motif_len'])))
	
	summary_file_name = output_prefix + ".p_val_len_summary"
	output_summary_file = open(summary_file_name, "w")
	
	for rank in range(min(max_rank, len(final_motifs))):
		print("printing summary for rank %d" % (rank))
		
		name_prefix = output_prefix + "_" + str(rank + 1)
		output_motif_as_range_file_name = name_prefix + ".motif_as_range"
		output_bed_file_name = name_prefix + ".bed"
		output_shape_file_name = name_prefix + ".shape"
		output_dist_file_name = name_prefix + ".dist.dat"
		output_extended_bed_file_name = name_prefix + ".extended.bed"

		output_motif_as_range_file = open(output_motif_as_range_file_name, "w")
		output_bed_file = open(output_bed_file_name, "w")
		output_shape_file = open(output_shape_file_name, "w")
		output_dist_file = open(output_dist_file_name, "w")
		output_extended_bed_file = open(output_extended_bed_file_name, "w")

		cur_motif = final_motifs[rank]
		motif_len = abs(cur_motif['motif_len'])
		

		'''
		(test_neglog_p_val, test_f_score, best_neglog_p_val, best_f_score, \
		 motif_id, motif_len, best_motif_as_range, best_k, \
		 best_tpr, best_fpr, test_tpr, test_fpr, \
		 best_chip_occurrences, test_chip_occurrences))
		'''
		for (min_val, max_val) in cur_motif['best_motif_as_range']:
			output_motif_as_range_file.write("%f\t%f\n" % (min_val, max_val))

		output_summary_file.write("%f\t%f\t%f\t%f\t%d\t%f\t%f\t%f\t%f\t%f\n" % \
		(cur_motif['best_neglog_p_val'], cur_motif['test_neglog_p_val'], \
		 cur_motif['best_f_score'], cur_motif['test_f_score'], \
		 abs(cur_motif['motif_len']), cur_motif['best_k'], \
		 cur_motif['best_tpr'], cur_motif['best_fpr'], \
		 cur_motif['test_tpr'], cur_motif['test_fpr']))

		extension = 50
		#use chip_shape_data, chip_bed_lines & test_chip_shape_data, test_chip_bed_lines
		training_occurrences = cur_motif['best_chip_occurrences']		
		for seq_index in training_occurrences:
			[chromosome, start_coord, end_coord] = chip_bed_lines[seq_index]
			cur_shape_data = chip_shape_data[seq_index]
			for location, strand_info in training_occurrences[seq_index]:
				loc_start = start_coord + location
				loc_end = loc_start + motif_len
				if strand_info == '-' and (feature_name == 'HelT' or feature_name == 'Roll'):
					output_bed_file.write("%s\t%d\t%d\t%s\n" % (chromosome, loc_start + 1, loc_end + 1, strand_info))
					output_dist_file.write("%d\n" % (location + 1))
				else:
					output_bed_file.write("%s\t%d\t%d\t%s\n" % (chromosome, loc_start, loc_end, strand_info))
					output_dist_file.write("%d\n" % (location))
					
				shape_profile = cur_shape_data[location : location + motif_len]
				if strand_info == '-':
					shape_profile = shape_profile[::-1]
				shape_profile_str = " ".join(map(str, shape_profile))
				output_shape_file.write("%s\n" % (shape_profile_str))

				extended_start = loc_start - extension
				extended_end = loc_end + extension
				if extended_start >= 2:
					output_extended_bed_file.write("%s\t%d\t%d\t%s\n" % (chromosome, extended_start, extended_end, strand_info))

		test_occurrences = cur_motif['test_chip_occurrences']
		for seq_index in test_occurrences:
			[chromosome, start_coord, end_coord] = test_chip_bed_lines[seq_index]
			cur_shape_data = test_chip_shape_data[seq_index]
			for location, strand_info in test_occurrences[seq_index]:
				loc_start = start_coord + location
				loc_end = loc_start + motif_len
				if strand_info == '-' and (feature_name == 'HelT' or feature_name == 'Roll'):
					output_bed_file.write("%s\t%d\t%d\t%s\n" % (chromosome, loc_start + 1, loc_end + 1, strand_info))
					output_dist_file.write("%d\n" % (location + 1))
				else:
					output_bed_file.write("%s\t%d\t%d\t%s\n" % (chromosome, loc_start, loc_end, strand_info))
					output_dist_file.write("%d\n" % (location))
					
				shape_profile = cur_shape_data[location : location + motif_len]
				if strand_info == '-':
					shape_profile = shape_profile[::-1]
				shape_profile_str = " ".join(map(str, shape_profile))
				output_shape_file.write("%s\n" % (shape_profile_str))

				extended_start = loc_start - extension
				extended_end = loc_end + extension
				if extended_start >= 2:
					output_extended_bed_file.write("%s\t%d\t%d\t%s\n" % (chromosome, extended_start, extended_end, strand_info))

		output_motif_as_range_file.close()
		output_bed_file.close()
		output_shape_file.close()
		output_dist_file.close()
		output_extended_bed_file.close()

	output_summary_file.close()

if __name__ == "__main__":
	sys.exit(main())
