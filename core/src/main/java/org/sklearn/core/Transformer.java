package org.sklearn.core;

/**
 * Interface for estimators that transform data.
 *
 * <p>Mirrors scikit-learn's {@code TransformerMixin}. Implementing classes
 * learn transformation parameters from training data via {@link #fit}
 * and apply the transformation via {@link #transform}.
 *
 * @param <D> the input data type
 * @param <T> the target/label type (may be {@link Void})
 */
public interface Transformer<D, T> extends Estimator<D, T> {

    /**
     * Fit the transformer and return the transformed data.
     *
     * <p>Convenience method equivalent to {@code fit(X, y).transform(X)}.
     *
     * @param X training samples, shape (n_samples, n_features)
     * @param y target values (may be null for unsupervised transformers)
     * @return transformed data
     */
    default D fitTransform(D X, T y) {
        fit(X, y);
        return transform(X);
    }

    /**
     * Apply the transformation to data X.
     *
     * @param X samples to transform, shape (n_samples, n_features)
     * @return transformed data
     */
    D transform(D X);

    /**
     * Reverse the transformation.
     *
     * @param X transformed samples, shape (n_samples, n_features)
     * @return data in the original space
     */
    D inverseTransform(D X);
}
