package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Standardize features by removing the mean and scaling to unit variance.
 *
 * <p>This transformer computes the mean and standard deviation for each feature
 * from the training data, then standardizes features by centering (mean removal)
 * and scaling to unit variance. This is the Java equivalent of
 * {@code sklearn.preprocessing.StandardScaler}.
 *
 * <p>Standardization uses population variance (ddof=0), matching the sklearn
 * default behavior. For features with near-zero standard deviation
 * ({@code < 1e-15}), the scale is set to 1.0 to avoid division by zero.
 *
 * <p>All operations are deterministic: repeated calls with the same input
 * produce identical results.
 */
public class StandardScaler implements Transformer<Matrix, Void> {

    private Vector mean;

    private Vector scale;

    private Vector var;

    private int nSamplesSeen;

    private boolean fitted;

    /**
     * Fit the scaler by computing the mean and standard deviation for each
     * feature in X.
     *
     * <p>The mean and population standard deviation (ddof=0) are computed
     * for every feature column. After fitting, the scaler can be used to
     * transform or inverse-transform data.
     *
     * @param X training samples, shape (n_samples, n_features)
     * @param y ignored (may be {@code null})
     * @return this fitted scaler
     */
    @Override
    public StandardScaler fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        int m = X.cols();
        nSamplesSeen = n;

        mean = computeMean(X, m);
        var = computeVariance(X, m, mean);
        scale = computeScale(var);

        fitted = true;
        return this;
    }

    /**
     * Standardize X by centering and scaling.
     *
     * <p>For each feature column, the transformation is:
     * {@code X_std = (X - mean) / scale}.
     *
     * @param X samples to transform, shape (n_samples, n_features)
     * @return standardized data (a new matrix; the input is not modified)
     * @throws IllegalStateException if the scaler is not fitted
     * @throws IllegalArgumentException if the feature count does not match
     */
    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "StandardScaler");
        Validation.checkMatrix(X, -1);
        if (X.cols() != mean.size()) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + mean.size()
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        int m = X.cols();
        double[][] result = new double[n][m];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = (X.get(i, j) - mean.get(j)) / scale.get(j);
            }
        }

        return new Matrix(result);
    }

    /**
     * Reverse the standardization transform.
     *
     * <p>For each feature column, the inverse transformation is:
     * {@code X = X_std * scale + mean}.
     *
     * @param X transformed samples, shape (n_samples, n_features)
     * @return data in the original space (a new matrix)
     * @throws IllegalStateException if the scaler is not fitted
     * @throws IllegalArgumentException if the feature count does not match
     */
    @Override
    public Matrix inverseTransform(Matrix X) {
        Validation.checkFitted(fitted, "StandardScaler");
        Validation.checkMatrix(X, -1);
        if (X.cols() != mean.size()) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + mean.size()
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        int m = X.cols();
        double[][] result = new double[n][m];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = X.get(i, j) * scale.get(j) + mean.get(j);
            }
        }

        return new Matrix(result);
    }

    /**
     * Return the fitted parameters as an unmodifiable map.
     *
     * <p>The returned map contains the following keys:
     * <ul>
     *   <li>{@code "mean"} &mdash; per-feature mean as a {@link Vector}</li>
     *   <li>{@code "scale"} &mdash; per-feature scaling factor as a {@link Vector}</li>
     *   <li>{@code "var"} &mdash; per-feature population variance as a {@link Vector}</li>
     *   <li>{@code "n_samples_seen"} &mdash; number of training samples as an {@link Integer}</li>
     * </ul>
     *
     * @return unmodifiable map of parameter name to value
     */
    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("mean", mean);
        params.put("scale", scale);
        params.put("var", var);
        params.put("n_samples_seen", nSamplesSeen);
        return Collections.unmodifiableMap(params);
    }

    /**
     * Check whether this scaler has been fitted to data.
     *
     * @return {@code true} if {@link #fit} has been called
     */
    public boolean isFitted() {
        return fitted;
    }

    // ---------------------------------------------------------------
    // Private helpers
    // ---------------------------------------------------------------

    private static Vector computeMean(Matrix X, int m) {
        int n = X.rows();
        double[] means = new double[m];
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += X.get(i, j);
            }
            means[j] = sum / n;
        }
        return new Vector(means);
    }

    private static Vector computeVariance(Matrix X, int m, Vector mean) {
        int n = X.rows();
        double[] variances = new double[m];
        for (int j = 0; j < m; j++) {
            double sumSq = 0.0;
            double mu = mean.get(j);
            for (int i = 0; i < n; i++) {
                double diff = X.get(i, j) - mu;
                sumSq += diff * diff;
            }
            variances[j] = sumSq / n;
        }
        return new Vector(variances);
    }

    private static Vector computeScale(Vector var) {
        int m = var.size();
        double[] scales = new double[m];
        for (int j = 0; j < m; j++) {
            double std = Math.sqrt(var.get(j));
            scales[j] = std < 1e-15 ? 1.0 : std;
        }
        return new Vector(scales);
    }
}
