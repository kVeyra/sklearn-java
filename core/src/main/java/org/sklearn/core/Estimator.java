package org.sklearn.core;

import java.util.Map;

/**
 * Base interface for all estimators in sklearn-java.
 *
 * <p>Inspired by scikit-learn's {@code BaseEstimator}, every estimator must
 * implement {@link #fit} and expose its configuration via {@link #getParameters()}.
 *
 * <p>All implementations must be deterministic: repeated calls with the same
 * data must produce identical results.
 *
 * @param <D> the input data type
 * @param <T> the target/label type
 */
public interface Estimator<D, T> {

    /**
     * Fit the model to the training data.
     *
     * @param X training samples, shape (n_samples, n_features)
     * @param y target values, shape (n_samples,)
     * @return this estimator (fitted)
     */
    Estimator<D, T> fit(D X, T y);

    /**
     * Return the estimator's internal configuration as a map of
     * parameter name to value. This mirrors sklearn's {@code get_params()}.
     *
     * @return unmodifiable view of the parameter map
     */
    Map<String, Object> getParameters();
}
