import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of numpy or sklearn e.g. np.random.randint for random initialization or sklearn.metrics.accuracy_score to calculate accuracy of a learnt model
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_latent, my_latent_updated etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here

def get_features_17(X, z_vals):
    """
    Computes the 17-bit Arbiter PUF embedding.
    X: shape (N, 16) challenges
    z_vals: shape (N,) latent middle bits
    """
    N = X.shape[0]
    X_17 = np.zeros((N, 17), dtype=int)
    X_17[:, :8] = X[:, :8]
    X_17[:, 8] = z_vals
    X_17[:, 9:] = X[:, 8:]
    
    # Standard embedding for Arbiter PUF: cumulative product of (1 - 2*c) from right to left
    return np.cumprod(1 - 2 * X_17[:, ::-1], axis=1)[:, ::-1]

def get_features_16(X):
    """
    Computes the 16-bit Arbiter PUF embedding.
    X: shape (N, 16) challenges
    """
    return np.cumprod(1 - 2 * X[:, ::-1], axis=1)[:, ::-1]

################################
# Non Editable Region Starting #
################################
def my_latent( X, y ):
################################
#  Non Editable Region Ending  #
################################
    N = X.shape[0]
    
    # Precompute features for both possible states of the latent variable
    phi0 = get_features_17(X, np.zeros(N, dtype=int))
    phi1 = get_features_17(X, np.ones(N, dtype=int))
    
    best_w, best_b, best_z = None, None, None
    best_acc = -1
    
    # Run multiple random initializations to avoid bad local minima ("unluckiness")
    for trial in range(5):
        z = np.random.randint(2, size=N)
        clf = LogisticRegression(C=10.0, fit_intercept=True, solver='liblinear')
        
        # Alternating Optimization (Hard-EM)
        for iteration in range(15):
            # 1. M-Step: Fix z, optimize w, b
            phi_curr = np.where(z[:, None] == 1, phi1, phi0)
            clf.fit(phi_curr, y)
            w = clf.coef_[0]
            b = clf.intercept_[0]
            
            # 2. E-Step: Fix w, b, optimize z
            score0 = phi0.dot(w) + b
            score1 = phi1.dot(w) + b
            
            margin0 = (2*y - 1) * score0
            margin1 = (2*y - 1) * score1
            
            # Winner-take-all update
            new_z = (margin1 > margin0).astype(int)
            
            if np.all(new_z == z):
                break # Converged!
            z = new_z
            
        # Track best model across random restarts based on training accuracy
        acc = clf.score(phi_curr, y)
        if acc > best_acc:
            best_acc = acc
            best_w = w
            best_b = b
            best_z = z
            
    return best_w, best_b, best_z

################################
# Non Editable Region Starting #
################################
def my_latent_updated( X, y ):
################################
#  Non Editable Region Ending  #
################################
    N = X.shape[0]
    
    # Precompute features
    phi0 = get_features_17(X, np.zeros(N, dtype=int))
    phi1 = get_features_17(X, np.ones(N, dtype=int))
    psi = get_features_16(X)
    
    best_w, best_b, best_u, best_a = None, None, None, None
    best_acc = -1
    
    # Run multiple random initializations
    for trial in range(5):
        z = np.random.randint(2, size=N)
        clf_r = LogisticRegression(C=10.0, fit_intercept=True, solver='liblinear')
        clf_z = LogisticRegression(C=10.0, fit_intercept=True, solver='liblinear')
        
        for iteration in range(15):
            phi_curr = np.where(z[:, None] == 1, phi1, phi0)
            
            # 1. M-Step: Fit 17-bit model predicting response r
            clf_r.fit(phi_curr, y)
            w = clf_r.coef_[0]
            b = clf_r.intercept_[0]
            
            # 2. M-Step: Fit 16-bit model predicting latent variable z
            if len(np.unique(z)) > 1:
                clf_z.fit(psi, z)
                u = clf_z.coef_[0]
                a = clf_z.intercept_[0]
            else:
                u = np.zeros(16)
                a = 0.0
            
            # 3. E-Step: Optimize z using joint likelihood
            score_r_0 = phi0.dot(w) + b
            score_r_1 = phi1.dot(w) + b
            
            # Log-likelihood terms for 17-bit response prediction
            log_pr_0 = -np.log1p(np.exp(-(2*y - 1) * score_r_0))
            log_pr_1 = -np.log1p(np.exp(-(2*y - 1) * score_r_1))
            
            # Log-likelihood terms for 16-bit latent bit prediction
            score_z = psi.dot(u) + a
            log_pz_0 = -np.log1p(np.exp(score_z))   # equivalent to ln(sigma(-score_z))
            log_pz_1 = -np.log1p(np.exp(-score_z))  # equivalent to ln(sigma(score_z))
            
            obj_0 = log_pr_0 + log_pz_0
            obj_1 = log_pr_1 + log_pz_1
            
            # Winner-take-all update based on joint probability
            new_z = (obj_1 > obj_0).astype(int)
            
            if np.all(new_z == z):
                break
            z = new_z
            
        acc = clf_r.score(phi_curr, y)
        if acc > best_acc:
            best_acc = acc
            best_w, best_b = w, b
            best_u, best_a = u, a
            
    return best_w, best_b, best_u, best_a;


