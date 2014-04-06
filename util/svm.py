def approximate_gamma(sample_matrix):
    """ Approximates the width parameter for the gaussian kernel.
        By computing the average distance between all training samples,
        we can approximate the width parameter of the gaussian and eliminate
        the need to optimize it through grid search.
    """
    return np.mean(-additive_chi2_kernel(sample_matrix))

def build_test_kernel(category, datamanager):
    X_train = datamanager.build_sample_matrix("train", category)
    X_test = datamanager.build_sample_matrix("test", category)
    gamma = approximate_gamma(X_train) # TODO: Wirklich X_train?
    kernel = chi2_kernel(X_test, X_train, gamma=1.0/gamma)
    return kernel, gamma
