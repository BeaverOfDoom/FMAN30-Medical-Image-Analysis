import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -------------------------------------------------
# Global settings
# -------------------------------------------------

np.random.seed(0)
m1, m2 = 0.0, 1.0
prior1 = prior2 = 0.5


# -------------------------------------------------
# Data generation
# -------------------------------------------------

def generate_data(smodel):
    x1_train = np.random.normal(m1, smodel, 10)
    x2_train = np.random.normal(m2, smodel, 10)
    x1_test  = np.random.normal(m1, smodel, 1000)
    x2_test  = np.random.normal(m2, smodel, 1000)
    return x1_train, x2_train, x1_test, x2_test


# -------------------------------------------------
# Parzen window KDE
# -------------------------------------------------

def parzen_estimate(x, train_data, h):
    p = np.zeros_like(x, dtype=float)
    for xi in train_data:
        p += norm.pdf(x, xi, h)
    return p / len(train_data)


# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def make_x_grid(x1, x2):
    return np.linspace(min(x1.min(), x2.min()) - 3,
                       max(x1.max(), x2.max()) + 3, 1000)


def true_density(x, mean, smodel):
    return norm.pdf(x, mean, smodel)


def plugin_predict(x, x1_train, x2_train, h):
    p1 = parzen_estimate(x, x1_train, h)
    p2 = parzen_estimate(x, x2_train, h)
    return np.where(p1 > p2, 1, 2)


def loocv_error(x1, x2, h):
    errors = 0
    total = 0

    for i in range(len(x1)):
        xi = x1[i]
        p1 = parzen_estimate(np.array([xi]), np.delete(x1, i), h)
        p2 = parzen_estimate(np.array([xi]), x2, h)
        errors += (1 if p1 < p2 else 0)
        total += 1

    for i in range(len(x2)):
        xi = x2[i]
        p1 = parzen_estimate(np.array([xi]), x1, h)
        p2 = parzen_estimate(np.array([xi]), np.delete(x2, i), h)
        errors += (1 if p2 < p1 else 0)
        total += 1

    return errors / total


# -------------------------------------------------
# Plotting helpers
# -------------------------------------------------

def plot_densities(x, p1_true, p2_true, p1_est, p2_est, x1, x2, smodel, sparzen):
    plt.figure(figsize=(8,5))
    plt.plot(x, p1_true, 'b--', label='p(x|y=1) true')
    plt.plot(x, p2_true, 'r--', label='p(x|y=2) true')
    plt.plot(x, p1_est, 'b', label='p(x|y=1) Parzen')
    plt.plot(x, p2_est, 'r', label='p(x|y=2) Parzen')
    plt.scatter(x1, np.zeros_like(x1), c='b')
    plt.scatter(x2, np.zeros_like(x2), c='r', marker='x')
    plt.title(f'smodel={smodel}, sparzen={sparzen}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_posteriors(x, post_true, post_est, smodel, sparzen):
    plt.figure(figsize=(8,5))
    plt.plot(x, post_true, 'k--', label='True posterior')
    plt.plot(x, post_est, 'k', label='Estimated posterior')
    plt.title(f'Posteriors: smodel={smodel}, sparzen={sparzen}')
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# Experiment runner
# -------------------------------------------------

def run_experiment(smodel, sparzen):
    x1_tr, x2_tr, x1_te, x2_te = generate_data(smodel)
    x = make_x_grid(x1_tr, x2_tr)

    p1_true = true_density(x, m1, smodel)
    p2_true = true_density(x, m2, smodel)
    p1_est  = parzen_estimate(x, x1_tr, sparzen)
    p2_est  = parzen_estimate(x, x2_tr, sparzen)

    post_true = p1_true / (p1_true + p2_true)
    post_est  = p1_est  / (p1_est  + p2_est + 1e-12)

    X_test = np.concatenate([x1_te, x2_te])
    y_test = np.concatenate([np.ones_like(x1_te), 2*np.ones_like(x2_te)])
    y_pred = plugin_predict(X_test, x1_tr, x2_tr, sparzen)

    test_error = np.mean(y_pred != y_test)

    t_star = 0.5
    bayes_error = 0.5 * (
        1 - norm.cdf(t_star, m1, smodel) +
        norm.cdf(t_star, m2, smodel)
    )

    print(f"smodel={smodel}, sparzen={sparzen}")
    print(f"  Test error: {test_error:.4f}")
    print(f"  Bayes error: {bayes_error:.4f}")

    plot_densities(x, p1_true, p2_true, p1_est, p2_est, x1_tr, x2_tr, smodel, sparzen)
    plot_posteriors(x, post_true, post_est, smodel, sparzen)


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":

    for smodel in [0.4, 4]:
        for sparzen in [0.02, 1.0]:
            run_experiment(smodel, sparzen)

        h_vals = np.linspace(0.05, 2, 30)
        x1_tr, x2_tr, _, _ = generate_data(smodel)
        cv_errs = [loocv_error(x1_tr, x2_tr, h) for h in h_vals]

        plt.figure(figsize=(6,4))
        plt.plot(h_vals, cv_errs, '-o')
        plt.title(f'Cross-validation (smodel={smodel})')
        plt.xlabel('h')
        plt.ylabel('LOOCV error')
        plt.tight_layout()
        plt.show()
