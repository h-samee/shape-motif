#!/usr/bin/python

import sys
import os
import random

import numpy as np
import math
from scipy import stats
from scipy.stats import rv_discrete

def distance(x, y):
	sq_sum = 0
	
	for i, x_val in enumerate(x):
		y_val = y[i]
		xy_diff = x_val - y_val
		sq_sum += xy_diff * xy_diff
	
	return sq_sum

def distances(x, y):
	'''
	this function assumes:
	x is the window from the current sequence
	and y is the window from the current pool
	the function will return:
	distances between x and y
	and between _x (= the reverse of x) and y
	'''
	d = 0
	_d = 0
	_x = x[::-1]
	for i, x_val in enumerate(x):
		_x_val = _x[i]
		y_val = y[i]
		_d += (_x_val - y_val) * (_x_val - y_val)
		d += (x_val - y_val) * (x_val - y_val)

	return (_d, d)

def gibbs_motif_extension_finder(shape_data, motif_len, seed_motif, extent):
	
	n_seq = len(shape_data)
	seq_len = len(shape_data[0])
	window_size = motif_len + extent

	##parameters specific to this function
	max_consecutive_iters_wo_improvement = 10
	n_consecutive_iters_wo_improvement = 0

	epsilon_factor_improvement = 1e-5

	##compute the range of locations within which an extended motif will be searched
	start_locs = [0] * n_seq
	end_locs = [seq_len - window_size] * n_seq
	
	##if seed has been specified for every sequence, then update the start_locs and end_locs
	if len(seed_motif) == n_seq: 
		for i in range(n_seq):
			if seed_motif[i] - window_size < 0:
				start_locs[i] = 0
			else:
				start_locs[i] = seed_motif[i] - window_size
			
			if seed_motif[i] + 2 * window_size >= seq_len:
				end_locs[i] = seq_len - window_size
			else:
				end_locs[i] = seed_motif[i] + window_size
	
	##generate initial window locations
	window_locs = []
	window_strands = []
	for i in range(n_seq):
		window_locs.append(random.randint(start_locs[i], end_locs[i]))
		#note: random.randint(a, b): returns a random integer N, s.t.: a <= N <= b
		window_strands.append(1)
		#note: we assume 1 = +ve strand, 0 = -ve strand
	
	all_pair_distance = float("inf")

	while True:
		cur_window_locs = window_locs[:]
		cur_window_strands = window_strands[:]

		#sample one candidate for each sequence
		for i in range(n_seq):
			cur_shape_data = shape_data[i]
			
			#we will maintain two parallel tables, for candidate index and for the score
			#initially, the score is not exponentiated; it is simply the negative of the sum of pairwise distances
			candidate_table = []
			score_table = []
			
			for j in range(start_locs[i], end_locs[i] + 1):
				cur_window = cur_shape_data[j : j + window_size]
				cur_distance_1 = 0
				cur_distance_0 = 0
				for k in range(n_seq):
					if k != i:
						solution_window = shape_data[k][cur_window_locs[k] : cur_window_locs[k] + window_size]
						if cur_window_strands[k] == 1:
							cur_distances = distances(cur_window, solution_window)
						else:
							cur_distances = distances(cur_window, solution_window[::-1])
						cur_distance_0 += cur_distances[0]
						cur_distance_1 += cur_distances[1]
						
				candidate_table.append((j, 1))
				score_table.append(-cur_distance_1)
				candidate_table.append((j, 0))
				score_table.append(-cur_distance_0)
				
			
			#now we find the best score, i.e., max_score = max in the score_table
			#and subtract max_score from each candidate's score
			#after subtraction, we exponentiate each score
			#these steps ensure that we do not run into numerical issues due division by zero
			#for example, say we have two windows and -error1 > -error2 (error1 < error2)
			#if both error1 & error2 are large, then both exp(-error1) and exp(-error2) = zero
			#and we run into numerical issues
			#under this scheme:
			#exp(-error1)/(exp(-error1) + exp(-error2))
			#=exp(0)/(exp(0) + exp(-error2 + error1))
			#=1/(1 + 0)
			#
			max_score = np.max(score_table)

			for score_index in range(len(score_table)):
				score_table[score_index] -= max_score
				score_table[score_index] = math.exp(score_table[score_index])
			
			score_sum = sum(score_table)
			#score_sum = 0	
			#for score_index in range(len(score_table)):
			#	score_sum += score_table[score_index]
			
			#normalize score_table to compute a probability distribution
			for score_index in range(len(score_table)):
				score_table[score_index] /= score_sum
			#sample
			[sample_index] = rv_discrete(values = (range(len(candidate_table)), score_table)).rvs(size = 1)
			cur_window_locs[i] = candidate_table[sample_index][0]
			cur_window_strands[i] = candidate_table[sample_index][1]
	
		#compute the current all-pair-distance -- how coherent/homogeneous are the shape features
		cur_all_pair_distance = 0
		for i in range(n_seq):
			for j in range(i + 1, n_seq):
				window_i = shape_data[i][cur_window_locs[i] : cur_window_locs[i] + window_size]
				if cur_window_strands[i] == 0:
					window_i = window_i[::-1]
				window_j = shape_data[j][cur_window_locs[j] : cur_window_locs[j] + window_size]
				if cur_window_strands[j] == 0:
					window_j = window_j[::-1]
				cur_all_pair_distance += distance(window_i, window_j)
		
		#update and continue, or break and terminate

		#if: no change in two successive iterations:
		if not (cur_all_pair_distance != all_pair_distance):
			window_locs = cur_window_locs[:]
			window_strands = cur_window_strands[:]
			break	
		
		#elif: we can assume that we have converged due to negligible changes in several successive iterations
		elif (not (cur_all_pair_distance > all_pair_distance)) and (cur_all_pair_distance > all_pair_distance * (1 - epsilon_factor_improvement)):
			all_pair_distance = cur_all_pair_distance
			window_locs = cur_window_locs[:]
			window_strands = cur_window_strands[:]
			n_consecutive_iters_wo_improvement += 1
			if n_consecutive_iters_wo_improvement == max_consecutive_iters_wo_improvement:
				break
		
		#else: our increase/decrease is too large to assume that we have converged
		else:
			all_pair_distance = cur_all_pair_distance
			window_locs = cur_window_locs[:]
			window_strands = cur_window_strands[:]
			n_consecutive_iters_wo_improvement = 0

	return window_locs, window_strands

def gibbs_motif_finder(shape_data, window_size):
	return gibbs_motif_extension_finder(shape_data, window_size, [], 0)


def compute_ranges(shape_data, motif_window_locs, window_size, sigma_count):
	n_seq = len(shape_data)
	ranges = []
	
	for i in range(window_size):
		vals_at_i = []
		for j in range(n_seq):
			data_val = shape_data[j][motif_window_locs[j] + i]
			vals_at_i.append(data_val)
		
		
		mean_at_i = np.mean(np.array(vals_at_i))
		std_at_i = np.std(np.array(vals_at_i))

		min_val = mean_at_i - sigma_count * std_at_i 
		max_val = mean_at_i + sigma_count * std_at_i 
		
		#min_val = np.min(np.array(vals_at_i))
		#max_val = np.max(np.array(vals_at_i))

		ranges.append((min_val, max_val))
	
	return ranges

def does_motif_match_window(motif_as_range, window):
	window_len = len(window)
	assert window_len == len(motif_as_range)

	for i in range(window_len):
		data_val = window[i]
		min_val = motif_as_range[i][0]
		max_val = motif_as_range[i][1]
		if data_val < min_val or data_val > max_val:
			return False

	return True 

def list_motif_occurrences(cur_shape_data, motif_as_range):
	seq_len = len(cur_shape_data)
	window_size = len(motif_as_range)
	occurrence_list = []
	for i in range(seq_len - window_size + 1):
		cur_window = cur_shape_data[i : i + window_size]
		if does_motif_match_window(motif_as_range, cur_window):
			occurrence_list.append(i)
	
	return occurrence_list

def count_occurrences(motif_as_range, shape_data):
	"""
	Note: 
	-- This function returns:
	---- number of sequences that contain the motif and
	---- locations of the motif's occurrences (overlapping occurrences allowed) in each sequence (as a dictionary)
	"""
	n_seq = len(shape_data)
	seq_len = len(shape_data[0])
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

def read_shape_data(file_name, max_lines):
	bed_lines = []
	shape_data = []
	data_size = 0
	with open(file_name) as f:
		for line in f:
			vals = line.split()
			if len(vals) != 5:
				print('Possible error in the .dat file of shape profile: some fields missing in data file integrating .bw and bed file')
			else:
				chromosome = vals[0]
				start_coord = int(vals[1])
				end_coord = int(vals[2])
				bed_lines.append([chromosome, start_coord, end_coord])

				data_len = int(vals[-2])
				data_vals = map(float, vals[-1].split(','))
				if len(data_vals) != data_len:
					print('Error: mismatch in bwtool generated data and data-len')
				else:
					shape_data.append(data_vals)
				data_size += 1
				if data_size == max_lines:
					break
	return shape_data, bed_lines

def main(argv = None):
	if argv is None:
		argv = sys.argv
	
	##input args
	chip_data_file_name = argv[1] 
	max_chip_data_size = int(argv[2])
	window_size = int(argv[3]) #length of motif
	max_iter = int(argv[4]) #max number of motifs	
	output_dir = argv[5]

	##read shape data and bed file
	#chip peaks (train)
	chip_shape_data, chip_bed_lines = read_shape_data(chip_data_file_name, max_chip_data_size)
	n_chip_seq = len(chip_shape_data)

	##set output files
	file_name_prefix = os.path.split(chip_data_file_name)[1].split('.')[0]
	#output file names	
	motif_instance_file_name = output_dir + "/" + file_name_prefix + ".instance"
	#open output files
	bufsize = 1 #Note: 0 means unbuffered, 1 means line buffered
	motif_instance_file = open(motif_instance_file_name, "w", bufsize)
	
	##initiate the random number generator based on the name of the output directory.
	##the output directory name contains cell type, TF name, shape feature, and length.
	random.seed(output_dir)

	for iter_count in range(max_iter):
		##compute motif
		motif_window_locs, motif_window_strands = gibbs_motif_finder(chip_shape_data, window_size)
		
		extended_motif_window_locs = [motif_window_locs]
		extended_motif_window_strands = [motif_window_strands]

		for extent in range(0,5):
			motif_window_locs, motif_window_strands = gibbs_motif_extension_finder(chip_shape_data, window_size, extended_motif_window_locs[0], extent)
			extended_motif_window_locs.append(motif_window_locs)
			extended_motif_window_strands.append(motif_window_strands)
		#extended_motif_window_locs here contains 6 lists:
		#first one is the original motif
		#second one is the shifted version of the original motif
		#the last 4 are the extended versions of the original motif
		len_offsets = [0, 0, 1, 2, 3, 4]		
		for motif_index in range(len(extended_motif_window_locs)):
			motif_window_locs = extended_motif_window_locs[motif_index]
			motif_window_strands = extended_motif_window_strands[motif_index]
			cur_motif_len = window_size + len_offsets.pop(0)
			##output motif instances
			motif_instance_file.write("#%d,%d\n" % (iter_count + 1, cur_motif_len))
			for i in range(n_chip_seq):
				cur_output_data = chip_shape_data[i][motif_window_locs[i] : motif_window_locs[i] + cur_motif_len]
				if motif_window_strands[i] == 0:
					cur_output_data_ = cur_output_data[::-1]
					cur_output_data = cur_output_data_
				for j in range(cur_motif_len):
					motif_instance_file.write("%f " % (cur_output_data[j]))
				motif_instance_file.write("\n")
	
	##close output files
	motif_instance_file.close()
	return 0

if __name__ == "__main__":
	sys.exit(main())
