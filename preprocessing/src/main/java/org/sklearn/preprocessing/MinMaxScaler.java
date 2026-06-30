package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Transform features by scaling each feature to a given range.
 *
 * <p>This estimator scales and translates each feature individually such
 * that it is in the given range on the training set, e.g. between zero
 * and one. This is the Java equivalent of
 * {@code sklearn.preprocessing.MinMaxScaler}.
 *
 * <p>The transformation is:
 * {@code X_std = (X - min) / (max - min)}
 * {@code X_scaled = X_std * (featureMax - featureMin) + featureMin}
 *
 * <p>For features with near-constant values ({@code max - min < 1e-15}),
 * the scale is set to 1.0 and the transformed value is mapped to
 * {@code featureMin}, matching the sklearn behavior.
 *
 * <p>All operations are deterministic: repeated calls with the same input
 * produce identical results.
 */
public class MinMaxScaler implements Transformer<Matrix, Void> {

    private Vector min;

    private Vector scale;

    private Vector dataMin;

    private Vector dataMax;

    private final double[] featureRange;

    private boolean fitted;

    /**
     * Construct a MinMaxScaler with default feature range {@code [0, 1]}.
     */
    public MinMaxScaler() {
        this(0.0, 1.0);
    }

    /**
     * Construct a MinMaxScaler with a custom feature range.
     *
     * @param featureMin the lower bound of the output range
     * @param featureMax the upper bound of the output range
     * @throws IllegalArgumentException if {@code featureMin >= featureMax}
     */
    public MinMaxScaler(double featureMin, double featureMax) {
        if (featureMin >= featureMax) {
            throw new IllegalArgumentException(
                "featureMin (" + featureMin + ") must be less than featureMax (" + featureMax + ")");
        }
        this.featureRange = new double[]{featureMin, featureMax};
    }

    /**
     * Fit the scaler by computing the minimum and maximum for each feature.
     *
     * <p>After fitting, the scaler stores the per-feature min, max, and scale
     * ({@code max - min}). Features with constant values are handled by
     * setting their scale to 1.0.
     *
     * @param X training samples, shape (n_samples, n_features)
     * @param y ignored (may be {@code null})
     * @return this fitted scaler
     */
    @Override
    public MinMaxScaler fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int m = X.cols();

        dataMin = computeMin(X, m);
        dataMax = computeMax(X, m);
        scale = computeScale(dataMin, dataMax);
        min = computeMinOffset(dataMin, scale, featureRange);

        fitted = true;
        return this;
    }

    /**
     * Scale X to the feature range.
     *
     * <p>For each feature column, the transformation is:
     * {@code X_scaled = (X - dataMin) / scale * (featureMax - featureMin) + featureMin}.
     *
     * @param X samples to transform, shape (n_samples, n_features)
     * @return scaled data (a new matrix; the input is not modified)
     * @throws IllegalStateException    if the scaler is not fitted
     * @throws IllegalArgumentException if the feature count does not match
     */
    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "MinMaxScaler");
        Validation.checkMatrix(X, -1);
        if (X.cols() != dataMin.size()) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + dataMin.size()
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        int m = X.cols();
        double fMin = featureRange[0];
        double fRange = featureRange[1] - featureRange[0];
        double[][] result = new double[n][m];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double xStd = (X.get(i, j) - dataMin.get(j)) / scale.get(j);
                result[i][j] = xStd * fRange + fMin;
            }
        }

        return new Matrix(result);
    }

    /**
     * Reverse the scaling transform back to the original feature space.
     *
     * <p>For each feature column, the inverse transformation is:
     * {@code X = (X_scaled - featureMin) / (featureMax - featureMin) * scale + dataMin}.
     *
     * @param X scaled samples, shape (n_samples, n_features)
     * @return data in the original space (a new matrix)
     * @throws IllegalStateException    if the scaler is not fitted
     * @throws IllegalArgumentException if the feature count does not match
     */
    @Override
    public Matrix inverseTransform(Matrix X) {
        Validation.checkFitted(fitted, "MinMaxScaler");
        Validation.checkMatrix(X, -1);
        if (X.cols() != dataMin.size()) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + dataMin.size()
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        int m = X.cols();
        double fMin = featureRange[0];
        double fRange = featureRange[1] - featureRange[0];
        double[][] result = new double[n][m];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = (X.get(i, j) - fMin) / fRange * scale.get(j) + dataMin.get(j);
            }
        }

        return new Matrix(result);
    }

    /**
     * Return the fitted parameters as an unmodifiable map.
     *
     * <p>The returned map contains the following keys:
     * <ul>
     *   <li>{@code "data_min"} &mdash; per-feature minimum as a {@link Vector}</li>
     *   <li>{@code "data_max"} &mdash; per-feature maximum as a {@link Vector}</li>
     *   <li>{@code "data_range"} &mdash; per-feature range as a {@link Vector}</li>
     *   <li>{@code "scale"} &mdash; per-feature scaling factor as a {@link Vector}</li>
     *   <li>{@code "min"} &mdash; per-feature adjusted minimum as a {@link Vector}</li>
     *   <li>{@code "feature_range"} &mdash; the output feature range as a {@code double[]}</li>
     * </ul>
     *
     * @return unmodifiable map of parameter name to value
     */
    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("data_min", dataMin);
        params.put("data_max", dataMax);
        params.put("data_range", dataMax.subtract(dataMin));
        params.put("scale", scale);
        params.put("min", min);
        params.put("feature_range", featureRange.clone());
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

    private static Vector computeMin(Matrix X, int m) {
        int n = X.rows();
        double[] mins = new double[m];
        for (int j = 0; j < m; j++) {
            double minVal = X.get(0, j);
            for (int i = 1; i < n; i++) {
                if (X.get(i, j) < minVal) {
                    minVal = X.get(i, j);
                }
            }
            mins[j] = minVal;
        }
        return new Vector(mins);
    }

    private static Vector computeMax(Matrix X, int m) {
        int n = X.rows();
        double[] maxs = new double[m];
        for (int j = 0; j < m; j++) {
            double maxVal = X.get(0, j);
            for (int i = 1; i < n; i++) {
                if (X.get(i, j) > maxVal) {
                    maxVal = X.get(i, j);
                }
            }
            maxs[j] = maxVal;
        }
        return new Vector(maxs);
    }

    private static Vector computeScale(Vector dataMin, Vector dataMax) {
        int m = dataMin.size();
        double[] scales = new double[m];
        for (int j = 0; j < m; j++) {
            double range = dataMax.get(j) - dataMin.get(j);
            scales[j] = range < 1e-15 ? 1.0 : range;
        }
        return new Vector(scales);
    }

    private static Vector computeMinOffset(Vector dataMin, Vector scale, double[] featureRange) {
        int m = dataMin.size();
        double fMin = featureRange[0];
        double fRange = featureRange[1] - featureRange[0];
        double[] minOffset = new double[m];
        for (int j = 0; j < m; j++) {
            minOffset[j] = fMin - dataMin.get(j) * fRange / scale.get(j);
        }
        return new Vector(minOffset);
    }
}
