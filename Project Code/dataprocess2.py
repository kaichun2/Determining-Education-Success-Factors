import os
import numpy as np
import csv
import matplotlib.pyplot as plt

data_file = 'data_folder/MERGED2013_14_PP.csv'

all_data = []
with open(data_file) as data_f:
	reader_f = csv.DictReader(data_f)
	for row in reader_f:
		all_data.append(row)	

indicator_to_check = ['MN_EARN_WNE_P10', 'ADM_RATE', 'SAT_AVG', 'AVGFACSAL', 'INEXPFTE', 'NPT4_PUB', 'NPT4_PRIV', 'PPTUG_EF']
indicator_mean = []
# Get the mean value of each indicators
for ind in indicator_to_check:
	count = 0
	mu_total = 0
	for row in all_data:
		if row[ind] != 'NULL' and row[ind] != 'PrivacySuppressed':
			count += 1
			mu_total += float(row[ind])
	indicator_mean.append(mu_total / count)

# replace all null with its mean 
for ind in range(len(indicator_to_check)):
	indicator_now = indicator_to_check[ind]
	for row in all_data:
		if row[indicator_now] == 'NULL' or row[indicator_now] == 'PrivacySuppressed':
			row[indicator_now] = indicator_mean[ind]
		else:
			row[indicator_now] = float(row[indicator_now])

indicator_to_use = ['ADM_RATE', 'SAT_AVG', 'AVGFACSAL', 'INEXPFTE', 'PPTUG_EF']
success_indicators = ['MN_EARN_WNE_P10']
# Put the usable value into an array
x_to_use = np.zeros((len(all_data),len(indicator_to_use)))
y_to_use = np.zeros(len(all_data)) 
for row in range(len(all_data)):
	for col in range(len(indicator_to_use)):
		x_to_use[row][col] = all_data[row][indicator_to_use[col]]
	y_to_use[row] = all_data[row][success_indicators[0]]

#####################################################################################################################
# # doing k means clustering for every year to see what common indicators each cluster shared compared to others
# # starting with k = 3 and descending order of years from 2010 to 2017
def k_means(k, x):
	# do k_means
	total_row = x.shape[0]
	total_col = x.shape[1]
	mu = []
	init_mu_index = []
	# initialize the mu
	while len(init_mu_index) < k:
		index = np.random.randint(total_row)
		# prevent same mean (may need to change)
		if index not in init_mu_index:
			init_mu_index.append(index)
			mu.append(x[index])
	
	it = 0
	previous_total_loss = None
	total_loss = None
	while previous_total_loss is None or total_loss - previous_total_loss < 0:
		# update c
		c = np.zeros(total_row)
		for cur_row in range(total_row):
			x_i = x[cur_row]
			first_v = x_i - mu[0]
			min_loss = np.dot(first_v, first_v)
			min_j = 0
			for j in range(k):
				v = x_i - mu[j]
				cur_loss = np.dot(v, v)
				if cur_loss < min_loss:
					min_j = j
					min_loss = cur_loss
			c[cur_row] = min_j
		# update mu
		for j in range(k):
			# numerator and denominator for mu_j
			mu_j_num = np.zeros(total_col)
			mu_j_den = 0
			for cur_row in range(total_row): 
				if c[cur_row] == j:
					mu_j_num += x[cur_row]
					mu_j_den += 1
			mu[j] = 0
			if mu_j_den != 0:
				mu[j] = 1 / mu_j_den * mu_j_num
		it += 1
		# get loss function 
		previous_total_loss = total_loss
		total_loss = 0
		for cur_row in range(total_row):
			x_i = x[cur_row]
			mu_j = mu[int(c[cur_row])]
			v = x_i - mu_j
			total_loss += np.dot(v, v)
	return c , mu, total_loss

# for finding the min loss using different seed
min_seed = 0
min_loss = None
for i in range(10):
	x = np.random.randint(2**31)
	np.random.seed(x)
	k_clus = 4
	cluster, cluster_mean, total_loss = k_means(k_clus, x_to_use)
	if min_loss == None:
		min_loss = total_loss
		min_seed = x
	elif min_loss > total_loss:
		min_seed = x
		min_loss = total_loss
print(min_seed)

np.random.seed(min_seed)
cluster, cluster_mean, total_loss = k_means(k_clus, x_to_use)
print('avg error is ', total_loss / x_to_use.shape[0])

avg_earning = [0 for i in range(k_clus)]
avg_earning_count = [0 for i in range(k_clus)]
for cur_country in range(len(cluster)):
	# j_th cluster
	j = int(cluster[cur_country])
	# Average earning
	avg_earning[j] += all_data[cur_country][success_indicators[0]]
	avg_earning_count[j] += 1
for i in range(k_clus):
	avg_earning[i] /= avg_earning_count[i] 

# indicator_to_use = ['ADM_RATE', 'SAT_AVG', 'AVGFACSAL', 'INEXPFTE', 'PPTUG_EF']
# output the result
for j in range(k_clus):
	print()
	print(avg_earning_count[j])
	print("Average earning after 10 yrs in cluster ", j, " is ", avg_earning[j])
	for ind in range(len(indicator_to_use)):
		print(indicator_to_use[ind], ' is ', cluster_mean[j][ind])
	print()

indicator_mean = indicator_mean[1:len(indicator_mean)]
# Apply PCA to map data into 2-dim space
# subtract by the mean
for i in range(x_to_use.shape[0]):
	for j in range(x_to_use.shape[1]):
		x_to_use[i][j] -= indicator_mean[j]
# Find  variance
ind_var = [0 for i in range(x_to_use.shape[1])]
for i in range(x_to_use.shape[0]):
	for j in range(x_to_use.shape[1]):
		ind_var[j] += np.power(float(x_to_use[i][j]), 2)
for i in range(len(ind_var)):
	ind_var[i] /= x_to_use.shape[0]
	ind_var[i] = np.sqrt(ind_var[i])
# divided by variance
for i in range(x_to_use.shape[0]):
	for j in range(x_to_use.shape[1]):
		x_to_use[i][j] /= ind_var[j]

sigma = 0
for i in range(x_to_use.shape[0]):
	v = x_to_use[i]
	sigma += np.outer(v, v)
sigma *= (1 / x_to_use.shape[0])
w, v = np.linalg.eig(sigma)
# need the first 2 eigenvectors (2-dim)
u = [v[:,0], v[:,1]]
y = np.zeros((x_to_use.shape[0], 2))
y_max = [0 for i in range(2)]
y_min = [0 for i in range(2)]
for i in range(x_to_use.shape[0]):
	for j in range(2):
		cur_u = u[j]
		y[i][j] = np.dot(cur_u, x_to_use[i])
		if y[i][j] > y_max[j]:
			y_max[j] = y[i][j]
		if y[i][j] < y_min[j]:
			y_min[j] = y[i][j] 
most_sig_j = 0
for j in range(len(avg_earning_count)):
	if avg_earning_count[j] == 23:
		most_sig_j = j
		break
# plot
for i in range(y.shape[0]):
	# jth cluster 
	j = cluster[i]
	if j == most_sig_j:
		print(all_data[i]['INSTNM'])
	if j == 0:
		plt.plot(y[i][0], y[i][1], 'bx')
	elif j == 1:
		plt.plot(y[i][0], y[i][1], 'rx')
	elif j == 2:
		plt.plot(y[i][0], y[i][1], 'gx')
	else:
		plt.plot(y[i][0], y[i][1], 'yx')
plt.xlabel('Projected y_0')
plt.ylabel('Projected y_1')
plt.show()