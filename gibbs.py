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

	#generate initial window locations
	window_locs = []
	for i in range(n_seq):
		window_locs.append(random.randint(0, seq_len - window_size))
		#random.randint(a, b): returns a random integer N, s.t.: a <= N <= b
	
	all_pair_distance = float("inf")

	while True:
		cur_window_locs = window_locs[:]
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
		cur_all_pair_distance = 0
		for i in range(n_seq):
			for j in range(i + 1, n_seq):
				window_i = shape_data[i][cur_window_locs[i] : cur_window_locs[i] + window_size]
				window_j = shape_data[j][cur_window_locs[j] : cur_window_locs[j] + window_size]
				cur_all_pair_distance += distance(window_i, window_j)
		if cur_all_pair_distance < all_pair_distance:
			all_pair_distance = cur_all_pair_distance
			window_locs = cur_window_locs[:]
		else:
			break

	return window_locs
	#random.jumpahead(n_seq)

def compute_ranges(shape_data, motif_window_locs, window_size):
	n_seq = len(shape_data)
	ranges = []
	
	for i in range(window_size):
		vals_at_i = []
		for j in range(n_seq):
			data_val = shape_data[j][motif_window_locs[j] + i]
			vals_at_i.append(data_val)
		
		
		mean_at_i = np.mean(np.array(vals_at_i))
		std_at_i = np.std(np.array(vals_at_i))

		min_val = mean_at_i - 2 * std_at_i 
		max_val = mean_at_i + 2 * std_at_i 
		
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

def does_seq_contain_motif(cur_shape_data, motif_as_range):
	seq_len = len(cur_shape_data)
	window_size = len(motif_as_range)
	
	for i in range(seq_len - window_size + 1):
		cur_window = cur_shape_data[i : i + window_size]
		if does_motif_match_window(motif_as_range, cur_window):
			return True
	
	return False

def count_occurrences(motif_as_range, shape_data):
	n_seq = len(shape_data)
	seq_len = len(shape_data[0])
	window_size = len(motif_as_range)
	
	occurrence_count = 0
	
	for i in range(n_seq):
		cur_shape_data = shape_data[i]
		if does_seq_contain_motif(cur_shape_data, motif_as_range):
			occurrence_count += 1

	return occurrence_count

def read_shape_data(file_name):
	bed_lines = []
	shape_data = []
	with open(file_name) as f:
		for line in f:
			vals = line.split()
			if len(vals) != 8:
				print('Possible error: some fields missing in data file intersecting .bw and Leslie data')
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
1. We assume there is no header line in the shape data or in the bed files
2. in the function count_occurrences, we allow overlapping occurrences
"""

def main(argv = None):
	if argv is None:
		argv = sys.argv
	
	#input args
	chip_data_file_name = argv[1] 
	ctrl_data_file_name = argv[2] 
	window_size = int(argv[3]) #length of motif
	max_iter = int(argv[4]) #max number of motifs	
	test_chip_data_file_name = argv[5]
	test_ctrl_data_file_name = argv[6]
	output_dir = argv[7]

	#####
	#read shape data
	chip_shape_data, chip_bed_lines = read_shape_data(chip_data_file_name)
	n_chip_seq = len(chip_shape_data)

	#read control shape data
	ctrl_shape_data, ctrl_bed_lines = read_shape_data(ctrl_data_file_name)
	n_ctrl_seq = len(ctrl_shape_data)

	#####
	#read test shape data
	test_chip_shape_data, test_chip_bed_lines = read_shape_data(test_chip_data_file_name)
	n_test_chip_seq = len(test_chip_shape_data)

	#read test control shape data
	test_ctrl_shape_data, test_ctrl_bed_lines = read_shape_data(test_ctrl_data_file_name)
	n_test_ctrl_seq = len(test_ctrl_shape_data)

	file_name_prefix = os.path.split(chip_data_file_name)[1].split('.')[0]
	summary_file_name = output_dir + "/" + file_name_prefix + ".summary"
	motif_shape_file_name = output_dir + "/" + file_name_prefix + ".shape"
	range_motif_file_name = output_dir + "/" + file_name_prefix + ".range_motif"
	motif_bed_file_name = output_dir + "/" + file_name_prefix + ".bed"

	summary_file = open(summary_file_name, "w")
	motif_shape_file = open(motif_shape_file_name, "w")
	range_motif_file = open(range_motif_file_name, "w")
	motif_bed_file = open(motif_bed_file_name, "w")
	
	for iter_count in range(max_iter):
		random.jumpahead(n_chip_seq * (1 + iter_count))
		#compute motif
		motif_window_locs = gibbs_motif_finder(chip_shape_data, window_size)
		
		#compute statistical significance
		motif_as_range = compute_ranges(chip_shape_data, motif_window_locs, window_size)

		#####
		n_chip_seq_with_shape = count_occurrences(motif_as_range, chip_shape_data)
		n_ctrl_seq_with_shape = count_occurrences(motif_as_range, ctrl_shape_data)

		total_seq = n_chip_seq + n_ctrl_seq
		n_seq_with_shape = n_chip_seq_with_shape + n_ctrl_seq_with_shape
		neg_log_p_val = -math.log10(stats.hypergeom.sf(n_chip_seq_with_shape - 1, total_seq, n_chip_seq, n_seq_with_shape))
		
		#####
		n_test_chip_seq_with_shape = count_occurrences(motif_as_range, test_chip_shape_data)
		n_test_ctrl_seq_with_shape = count_occurrences(motif_as_range, test_ctrl_shape_data)

		total_test_seq = n_test_chip_seq + n_test_ctrl_seq
		n_test_seq_with_shape = n_test_chip_seq_with_shape + n_test_ctrl_seq_with_shape
		neg_log_p_val_test = -math.log10(stats.hypergeom.sf(n_test_chip_seq_with_shape - 1, total_test_seq, n_test_chip_seq, n_test_seq_with_shape))

		#generate outputs
		#summary: p-values
		summary_file.write("motif: %d\tneg_log_p_val: %f\tn_chip_w_shp: %d\tn_ctrl_w_shp: %d" %\
		 (iter_count + 1, neg_log_p_val, n_chip_seq_with_shape, n_ctrl_seq_with_shape))

		summary_file.write("\tneg_log_p_val_test: %f\tn_test_chip_w_shp: %d\tn_test_ctrl_w_shp: %d\n" %\
		 (neg_log_p_val_test, n_test_chip_seq_with_shape, n_test_ctrl_seq_with_shape))

		#shapes at motif occurrences
		#NOTE: this list may not contain a window from every sequence 
		motif_shape_file.write("#%d\n" % (iter_count + 1))
		for i in range(n_chip_seq):
			cur_window = chip_shape_data[i][motif_window_locs[i] : motif_window_locs[i] + window_size]
			if does_motif_match_window(motif_as_range, cur_window):
				for j in range(window_size):
					motif_shape_file.write("%f " % (chip_shape_data[i][motif_window_locs[i] + j]))
				motif_shape_file.write("\n")
		
		#range motif
		range_motif_file.write("#%d\n" % (iter_count + 1))
		for i in range(window_size):
			range_motif_file.write("%f " % (motif_as_range[i][0]))
		range_motif_file.write("\n")
		for i in range(window_size):
			range_motif_file.write("%f " % (motif_as_range[i][1]))
		range_motif_file.write("\n")

		#bed coordiates at motif occurrences
		#NOTE: this list may not contain a line for every sequence 
		motif_bed_file.write("#%d\n" % (iter_count + 1))
		for i in range(n_chip_seq):
			cur_window = chip_shape_data[i][motif_window_locs[i] : motif_window_locs[i] + window_size]
			if does_motif_match_window(motif_as_range, cur_window):
				[chromosome, start_coord, end_coord] = chip_bed_lines[i]
				motif_start_coord = start_coord + motif_window_locs[i]
				motif_end_coord = motif_start_coord + window_size
				motif_bed_file.write("%s\t%d\t%d\n" % (chromosome, motif_start_coord, motif_end_coord))
	
	summary_file.close()
	motif_bed_file.close()
	motif_shape_file.close()
	range_motif_file.close()
	return 0

if __name__ == "__main__":
	sys.exit(main())
