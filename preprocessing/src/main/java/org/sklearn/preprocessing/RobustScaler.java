package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Scale features using statistics that are robust to outliers.
 *
 * <p>Centers using the median and scales using the interquartile range (IQR).
 * The IQR is the range between the 1st quartile (25th percentile) and the
 * 3rd quartile (75th percentile). Features whose IQR is near-zero are left
 * unscaled to avoid division by zero.
 *
 * <p>Mirrors {@code sklearn.preprocessing.RobustScaler}.
 */
public class RobustScaler implements Transformer<Matrix, Void> {

    private Vector center;
    private Vector scale;
    private boolean fitted;

    @Override
    public RobustScaler fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int m = X.cols();
        center = new Vector(m);
        scale = new Vector(m);

        for (int j = 0; j < m; j++) {
            double[] col = new double[X.rows()];
            for (int i = 0; i < X.rows(); i++) {
                col[i] = X.get(i, j);
            }
            Arrays.sort(col);
            int n = col.length;
            center.set(j, percentile(col, 0.5));
            double q1 = percentile(col, 0.25);
            double q3 = percentile(col, 0.75);
            double iqr = q3 - q1;
            scale.set(j, iqr < 1e-15 ? 1.0 : iqr);
        }

        fitted = true;
        return this;
    }

    private static double percentile(double[] sorted, double p) {
        int n = sorted.length;
        double idx = p * (n - 1);
        int lo = (int) Math.floor(idx);
        int hi = (int) Math.ceil(idx);
        if (lo == hi) {
            return sorted[lo];
        }
        return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "RobustScaler");
        Validation.checkMatrix(X, -1);
        if (X.cols() != center.size()) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + center.size()
                    + " features, got " + X.cols());
        }

        int n = X.rows(), m = X.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = (X.get(i, j) - center.get(j)) / scale.get(j);
            }
        }
        return new Matrix(result);
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        Validation.checkFitted(fitted, "RobustScaler");
        Validation.checkMatrix(X, -1);
        if (X.cols() != center.size()) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + center.size()
                    + " features, got " + X.cols());
        }

        int n = X.rows(), m = X.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = X.get(i, j) * scale.get(j) + center.get(j);
            }
        }
        return new Matrix(result);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("center", center);
        params.put("scale", scale);
        return Collections.unmodifiableMap(params);
    }

    public boolean isFitted() {
        return fitted;
    }
}
