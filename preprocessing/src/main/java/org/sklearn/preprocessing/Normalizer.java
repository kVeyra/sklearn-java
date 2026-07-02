package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.utils.Validation;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Normalize samples individually to unit norm.
 *
 * <p>Each sample (i.e., each row of the data matrix) is scaled independently
 * so that its norm equals 1. This is a row-wise operation and requires no
 * training data, so {@link #fit} is a no-op that simply marks the instance
 * as fitted.
 *
 * <p>Three normalization strategies are supported:
 * <ul>
 *   <li><b>L2</b> &mdash; Euclidean norm (default)
 *   <li><b>L1</b> &mdash; sum of absolute values
 *   <li><b>MAX</b> &mdash; maximum absolute value
 * </ul>
 *
 * <p>Rows with a norm of zero are left unchanged. This is the Java equivalent
 * of {@code sklearn.preprocessing.Normalizer}.
 */
public class Normalizer implements Transformer<Matrix, Void> {

    /**
     * The type of norm to apply when normalizing samples.
     */
    public enum Norm {
        L1,
        L2,
        MAX
    }

    private final Norm norm;

    private boolean fitted;

    /**
     * Construct a Normalizer with the default L2 norm.
     */
    public Normalizer() {
        this(Norm.L2);
    }

    /**
     * Construct a Normalizer with the specified norm.
     *
     * @param norm the normalization strategy
     */
    public Normalizer(Norm norm) {
        if (norm == null) {
            throw new IllegalArgumentException("Norm must not be null");
        }
        this.norm = norm;
    }

    /**
     * No-op fit method.
     *
     * <p>Normalizer does not need to learn any parameters from data.
     * This method simply validates the input and marks the instance as fitted.
     *
     * @param X training samples, shape (n_samples, n_features)
     * @param y ignored (may be {@code null})
     * @return this fitted normalizer
     */
    @Override
    public Normalizer fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        fitted = true;
        return this;
    }

    /**
     * Normalize each sample (row) to unit norm.
     *
     * <p>For each row the norm is computed according to the selected
     * {@link Norm} type, and every element in the row is divided by
     * that norm. Rows with a norm of zero are copied unchanged.
     *
     * @param X samples to normalize, shape (n_samples, n_features)
     * @return normalized data (a new matrix; the input is not modified)
     * @throws IllegalStateException if the normalizer is not fitted
     */
    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "Normalizer");
        Validation.checkMatrix(X, -1);

        int n = X.rows();
        int m = X.cols();
        double[][] result = new double[n][m];

        for (int i = 0; i < n; i++) {
            double normValue = computeRowNorm(X, i, m);
            if (normValue == 0.0) {
                for (int j = 0; j < m; j++) {
                    result[i][j] = X.get(i, j);
                }
            } else {
                double inv = 1.0 / normValue;
                for (int j = 0; j < m; j++) {
                    result[i][j] = X.get(i, j) * inv;
                }
            }
        }

        return new Matrix(result);
    }

    /**
     * Throws {@link UnsupportedOperationException}.
     *
     * <p>Normalization is not invertible in general, so inverse transform
     * is not supported.
     *
     * @param X transformed samples (ignored)
     * @return never returns normally
     * @throws UnsupportedOperationException always
     */
    @Override
    public Matrix inverseTransform(Matrix X) {
        throw new UnsupportedOperationException(
            "Normalizer does not support inverseTransform");
    }

    /**
     * Return the parameters as an unmodifiable map.
     *
     * <p>The returned map contains the following key:
     * <ul>
     *   <li>{@code "norm"} &mdash; the norm type as a {@link String}</li>
     * </ul>
     *
     * @return unmodifiable map of parameter name to value
     */
    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("norm", norm.name());
        return Collections.unmodifiableMap(params);
    }

    /**
     * Check whether this normalizer has been fitted.
     *
     * @return {@code true} if {@link #fit} has been called
     */
    public boolean isFitted() {
        return fitted;
    }

    /**
     * Return the norm type used by this normalizer.
     *
     * @return the norm
     */
    public Norm getNorm() {
        return norm;
    }

    // ---------------------------------------------------------------
    // Private helpers
    // ---------------------------------------------------------------

    private double computeRowNorm(Matrix X, int row, int cols) {
        switch (norm) {
            case L1:
                double sum = 0.0;
                for (int j = 0; j < cols; j++) {
                    sum += Math.abs(X.get(row, j));
                }
                return sum;
            case L2:
                double sumSq = 0.0;
                for (int j = 0; j < cols; j++) {
                    double v = X.get(row, j);
                    sumSq += v * v;
                }
                return Math.sqrt(sumSq);
            case MAX:
                double max = 0.0;
                for (int j = 0; j < cols; j++) {
                    double abs = Math.abs(X.get(row, j));
                    if (abs > max) {
                        max = abs;
                    }
                }
                return max;
            default:
                throw new AssertionError("Unknown norm: " + norm);
        }
    }
}
