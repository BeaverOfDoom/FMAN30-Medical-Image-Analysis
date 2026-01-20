import numpy as np
import matplotlib.pyplot as plt


# Convert complex 
def complex_to_vec(shape_cpx):
    x = shape_cpx.real
    y = shape_cpx.imag
    return np.vstack((x, y)).T.flatten()


# Convert vector back to complex shape
def vec_to_complex(vec):
    x = vec[0::2]
    y = vec[1::2]
    return x + 1j*y



# Build PCA model from aligned shapes
def build_shape_model(aligned_shapes):
   
    N_points, N_shapes = aligned_shapes.shape
    D = 2 * N_points  

    # Convert each shape to 28D vector
    X = np.zeros((D, N_shapes))
    for i in range(N_shapes):
        X[:, i] = complex_to_vec(aligned_shapes[:, i])

    # Mean vector
    mean_vec = np.mean(X, axis=1, keepdims=True)

    # Center data
    Xc = X - mean_vec

    # Covariance matrix
    C = (Xc @ Xc.T) / (N_shapes)

    # PCA: eigen decomposition
    vals, vecs = np.linalg.eigh(C)

    # Sort by descending eigenvalue
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    return mean_vec.flatten(), vecs, vals



# Plot eigenvalues
def plot_eigenvalues(vals):
    plt.figure()
    plt.plot(vals, "o-")
    plt.title("Eigenvalues of Shape Model")
    plt.xlabel("Mode number")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()



# Plot PCA modes: mean +- k * sqrt(lambda)
def show_modes(mean_vec, vecs, vals, num_modes=8, k=2):
    N = len(mean_vec) // 2

    for mode in range(num_modes):
        eigval = vals[mode]
        eigvec = vecs[:, mode]

        # Mode shapes
        plus = mean_vec + k * np.sqrt(eigval) * eigvec
        minus = mean_vec - k * np.sqrt(eigval) * eigvec

        plus_c = vec_to_complex(plus)
        minus_c = vec_to_complex(minus)
        mean_c = vec_to_complex(mean_vec)

        plt.figure(figsize=(6,6))
        plt.plot(mean_c.real, mean_c.imag, "ko-", label="Mean")
        plt.plot(plus_c.real, plus_c.imag, "r.-", label="+mode")
        plt.plot(minus_c.real, minus_c.imag, "b.-", label="-mode")

        for i in range(N):
            plt.plot([mean_c.real[i], plus_c.real[i]],
                     [mean_c.imag[i], plus_c.imag[i]], "r--", alpha=0.3)
            plt.plot([mean_c.real[i], minus_c.real[i]],
                     [mean_c.imag[i], minus_c.imag[i]], "b--", alpha=0.3)

        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.title(f"Mode {mode+1}")
        plt.legend()
        plt.show()