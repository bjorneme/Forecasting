import numpy as np

np.random.seed(0)  # For reproducibility

# Assume we have a hypothetical user-item interaction matrix R
R = np.array([
    [5, 3, 0],
    [4, 0, 0],
    [1, 1, 0]
])

num_users, num_items = R.shape
num_factors = 3  # Assuming the number of latent factors is 3 for simplicity

# Initialize U (user matrix) and I (item matrix) with random values
U = np.random.rand(num_users, num_factors)
I = np.random.rand(num_items, num_factors)

# ALS parameters
num_iterations = 10  # Number of iterations
λ = 0.1  # Regularization parameter to prevent overfitting

for iteration in range(num_iterations):
    # Update user matrix U
    for i in range(num_users):
        U[i] = np.linalg.solve(np.dot(I.T, I) + λ * np.eye(num_factors), np.dot(I.T, R[i].T)).T
    # Update item matrix I
    for j in range(num_items):
        I[j] = np.linalg.solve(np.dot(U.T, U) + λ * np.eye(num_factors), np.dot(U.T, R[:, j]))

# Compute the predicted user-item interaction matrix after ALS optimization
predicted_R = np.dot(U, I.T)

# Output the updated matrices and the new predicted user-item interaction matrix
print("Updated User Matrix (U) after ALS:")
print(U)
print("\nUpdated Item Matrix (I) after ALS:")
print(I)
print("\nNew Predicted User-Item Interaction Matrix after ALS:")
print(predicted_R)
