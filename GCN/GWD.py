import numpy as np
import ot

def compute_gw_distance(C1, C2, p, q):
    # The Gromov-Wasserstein distance is a specific case of the 
    # more general Gromov-Wasserstein discrepancy. Here, we
    # use the 'square_loss' loss function.
    gw_dist = ot.gromov_wasserstein(C1, C2, p, q, 'square_loss')

    return gw_dist

# Define the distance matrices for three simple graphs
C1 = np.random.rand(3, 3)
C2 = np.random.rand(3, 3)
C3 = np.random.rand(3, 3)

# Normalize the matrices to make them proper distance matrices
C1 /= C1.max()
C2 /= C2.max()
C3 /= C3.max()

# Define the distributions of the nodes in the three graphs
p = np.ones(3) / 3
q = np.ones(3) / 3
r = np.ones(3) / 3

# Compute the Gromov-Wasserstein distance between each pair of graphs
gw_dist_12 = compute_gw_distance(C1, C2, p, q)
gw_dist_13 = compute_gw_distance(C1, C3, p, r)
gw_dist_23 = compute_gw_distance(C2, C3, q, r)
print('graph 1', C1,p)
print('graph 2', C2,q)
print('graph 3', C3,r)
# Print the results
print('The Gromov-Wasserstein distance between graph 1 and 2 is:', gw_dist_12)
print('The Gromov-Wasserstein distance between graph 1 and 3 is:', gw_dist_13)
print('The Gromov-Wasserstein distance between graph 2 and 3 is:', gw_dist_23)

# Find the minimum distance and the corresponding pair of graphs
distances = [gw_dist_12, gw_dist_13, gw_dist_23]
pair_names = ['1 and 2', '1 and 3', '2 and 3']
min_index = np.argmin(distances)
print('The most similar pair of graphs is:', pair_names[min_index])
