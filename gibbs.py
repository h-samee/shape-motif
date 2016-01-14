#!/usr/bin/python

import sys
import os
import random

import numpy as np
import math
from scipy import stats


def distance(x, y):
	sq_sum = 0
	
	for i, x_val in enumerate(x):
		y_val = y[i]
		xy_diff = x_val - y_val
		sq_sum += xy_diff * xy_diff
	
	return sq_sum

def gibbs_motif_finder(shape_data, window_size):
	n_seq = len(shape_data)
	seq_len = len(shape_data[0])
	##parameters specific to this function
	max_consecutive_iters_wo_improvement = 10
	n_consecutive_iters_wo_improvement = 0
	epsilon_factor_improvement = 1e-5

	##generate initial window locations
	window_locs = []
	for i in range(n_seq):
		window_locs.append(random.randint(0, seq_len - window_size))
		#random.randint(a, b): returns a random integer N, s.t.: a <= N <= b
	
	all_pair_distance = float("inf")

	while True:
		cur_window_locs = window_locs[:]
		#find the current best candidate for each sequence
		for i in range(n_seq):
			cur_shape_data = shape_data[i]
			cur_best_distance = float("inf")
			for j in range(seq_len - window_size + 1):
				cur_window = cur_shape_data[j : j + window_size]
				cur_distance = 0
				for k in range(n_seq):
					if k != i:
						solution_window = shape_data[k][cur_window_locs[k] : cur_window_locs[k] + window_size]
						cur_distance += distance(cur_window, solution_window)
				if cur_distance < cur_best_distance:
					cur_best_distance = cur_distance
					cur_window_locs[i] = j
		#compute the current all-pair-distance -- how coherent/homogeneous are the shape features
		cur_all_pair_distance = 0
		for i in range(n_seq):
			for j in range(i + 1, n_seq):
				window_i = shape_data[i][cur_window_locs[i] : cur_window_locs[i] + window_size]
				window_j = shape_data[j][cur_window_locs[j] : cur_window_locs[j] + window_size]
				cur_all_pair_distance += distance(window_i, window_j)
		
		#update and continue, or break and terminate
		if cur_all_pair_distance > all_pair_distance:
			break
		elif cur_all_pair_distance > all_pair_distance * (1 - epsilon_factor_improvement):
			all_pair_distance = cur_all_pair_distance
			window_locs = cur_window_locs[:]
			n_consecutive_iters_wo_improvement += 1
			if n_consecutive_iters_wo_improvement == max_consecutive_iters_wo_improvement:
				break
		else:
			all_pair_distance = cur_all_pair_distance
			window_locs = cur_window_locs[:]
			n_consecutive_iters_wo_improvement = 0

	return window_locs

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

def read_shape_data(file_name):
	bed_lines = []
	shape_data = []
	with open(file_name) as f:
		for line in f:
			vals = line.split()
			if len(vals) != 8:
				print('Possible error: some fields missing in data file integrating .bw and Leslie data')
			else:
				chromosome = vals[0]
				start_coord = int(vals[1])
				end_coord = int(vals[2])
				bed_lines.append([chromosome, start_coord, end_coord])

				data_len = int(vals[6])
				data_vals = map(float, vals[7].split(','))
				if len(data_vals) != data_len:
					print('Error: mismatch in bwtool generated data and data-len')
				else:
					shape_data.append(data_vals)
	return shape_data, bed_lines
"""
Notes:
-- This version takes inputs for the Arvey et al. paper (Leslie Lab).
---- Training (chip and ctrl peaks) and test (training and ctrl peaks)
---- window size
---- Number of motifs to find
---- output directory
-- We assume there is no header line in the shape data or in the bed files
-- in the function count_occurrences, we allow overlapping occurrences
"""

def main(argv = None):
	if argv is None:
		argv = sys.argv
	
	##input args
	chip_data_file_name = argv[1] 
	ctrl_data_file_name = argv[2] 
	window_size = int(argv[3]) #length of motif
	max_iter = int(argv[4]) #max number of motifs	
	test_chip_data_file_name = argv[5]
	test_ctrl_data_file_name = argv[6]
	output_dir = argv[7]

	##read shape data and bed file
	#chip peaks (train)
	chip_shape_data, chip_bed_lines = read_shape_data(chip_data_file_name)
	n_chip_seq = len(chip_shape_data)
	#control peaks (train)
	ctrl_shape_data, ctrl_bed_lines = read_shape_data(ctrl_data_file_name)
	n_ctrl_seq = len(ctrl_shape_data)
	#chip peaks (test)
	test_chip_shape_data, test_chip_bed_lines = read_shape_data(test_chip_data_file_name)
	n_test_chip_seq = len(test_chip_shape_data)
	#control peaks (train)
	test_ctrl_shape_data, test_ctrl_bed_lines = read_shape_data(test_ctrl_data_file_name)
	n_test_ctrl_seq = len(test_ctrl_shape_data)

	##set output files
	file_name_prefix = os.path.split(chip_data_file_name)[1].split('.')[0]
	#output file names	
	motif_instance_file_name = output_dir + "/" + file_name_prefix + ".instance"
	summary_file_name = output_dir + "/" + file_name_prefix + ".summary"
	motif_shape_file_name = output_dir + "/" + file_name_prefix + ".shape"
	motif_bed_file_name = output_dir + "/" + file_name_prefix + ".bed"
	#open output files
	bufsize = 1 #Note: 0 means unbuffered, 1 means line buffered
	motif_instance_file = open(motif_instance_file_name, "w", bufsize)
	summary_file = open(summary_file_name, "w", bufsize)
	motif_shape_file = open(motif_shape_file_name, "w", bufsize)
	motif_bed_file = open(motif_bed_file_name, "w", bufsize)
	
	##initiate the random number generator based on the name of the output directory.
	##the output directory name contains cell type, TF name, shape feature, and length.
	random.seed(output_dir)

	for iter_count in range(max_iter):
		##compute motif
		motif_window_locs = gibbs_motif_finder(chip_shape_data, window_size)
		
		##output motif instances
		for i in range(n_chip_seq):
			for j in range(window_size):
				motif_instance_file.write("%f " % (chip_shape_data[i][motif_window_locs[i] + j]))
			motif_instance_file.write("\n")

		for sigma_count in [0.5, 1, 1.5, 2]:
			##compute statistical significance
			motif_as_range = compute_ranges(chip_shape_data, motif_window_locs, window_size, sigma_count)

			#significance for training set
			n_chip_seq_with_shape, chip_occurrences = count_occurrences(motif_as_range, chip_shape_data)
			n_ctrl_seq_with_shape, ctrl_occurrences = count_occurrences(motif_as_range, ctrl_shape_data)

			total_seq = n_chip_seq + n_ctrl_seq
			n_seq_with_shape = n_chip_seq_with_shape + n_ctrl_seq_with_shape
			neg_log_p_val = -math.log10(stats.hypergeom.sf(n_chip_seq_with_shape - 1, total_seq, n_chip_seq, n_seq_with_shape))
			
			#significance for test set
			n_test_chip_seq_with_shape, test_chip_occurrences = count_occurrences(motif_as_range, test_chip_shape_data)
			n_test_ctrl_seq_with_shape, test_ctrl_occurrences = count_occurrences(motif_as_range, test_ctrl_shape_data)

			total_test_seq = n_test_chip_seq + n_test_ctrl_seq
			n_test_seq_with_shape = n_test_chip_seq_with_shape + n_test_ctrl_seq_with_shape
			neg_log_p_val_test = -math.log10(stats.hypergeom.sf(n_test_chip_seq_with_shape - 1, total_test_seq, n_test_chip_seq, n_test_seq_with_shape))

			##generate outputs
			#summary: p-values
			summary_file.write("motif:%d;sigma:%f;neg_log_p_val:%f;n_chip_w_shp:%d;n_ctrl_w_shp:%d" %\
			 (iter_count + 1, sigma_count, neg_log_p_val, n_chip_seq_with_shape, n_ctrl_seq_with_shape))

			summary_file.write(";neg_log_p_val_test:%f;n_test_chip_w_shp:%d;n_test_ctrl_w_shp:%d\n" %\
			 (neg_log_p_val_test, n_test_chip_seq_with_shape, n_test_ctrl_seq_with_shape))

			##shapes at and bed coordinates of motif occurrences
			#NOTE: this list may not contain a window from every sequence 
			motif_shape_file.write("#%d;%f\n" % (iter_count + 1, sigma_count))
			motif_bed_file.write("#%d;%f\n" % (iter_count + 1, sigma_count))

			for i in range(n_chip_seq):
				lst = chip_occurrences[i]
				[chromosome, start_coord, end_coord] = chip_bed_lines[i]
				for j in lst:
					for k in range(window_size):
						motif_shape_file.write("%f " % (chip_shape_data[i][j + k]))
					motif_shape_file.write("\n")

					motif_start_coord = start_coord + j
					motif_end_coord = motif_start_coord + window_size
					motif_bed_file.write("%s\t%d\t%d\n" % (chromosome, motif_start_coord, motif_end_coord))

	##close output files
	summary_file.close()
	motif_bed_file.close()
	motif_shape_file.close()
	return 0

if __name__ == "__main__":
	sys.exit(main())
