package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Scale each feature by its maximum absolute value.
 *
 * <p>This transformer scales and translates each feature individually such
 * that the maximal absolute value of each feature in the training set is 1.0.
 * It does not shift/center the data, and thus does not destroy sparsity.
 *
 * <p>Mirrors {@code sklearn.preprocessing.MaxAbsScaler}.
 */
public class MaxAbsScaler implements Transformer<Matrix, Void> {

    private Vector maxAbs;
    private boolean fitted;

    @Override
    public MaxAbsScaler fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int n = X.rows(), m = X.cols();
        maxAbs = new Vector(m);
        for (int j = 0; j < m; j++) {
            double max = 0.0;
            for (int i = 0; i < n; i++) {
                double abs = Math.abs(X.get(i, j));
                if (abs > max) {
                    max = abs;
                }
            }
            maxAbs.set(j, max < 1e-15 ? 1.0 : max);
        }
        fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "MaxAbsScaler");
        Validation.checkMatrix(X, -1);
        if (X.cols() != maxAbs.size()) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + maxAbs.size()
                    + " features, got " + X.cols());
        }

        int n = X.rows(), m = X.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = X.get(i, j) / maxAbs.get(j);
            }
        }
        return new Matrix(result);
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        Validation.checkFitted(fitted, "MaxAbsScaler");
        Validation.checkMatrix(X, -1);
        if (X.cols() != maxAbs.size()) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + maxAbs.size()
                    + " features, got " + X.cols());
        }

        int n = X.rows(), m = X.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = X.get(i, j) * maxAbs.get(j);
            }
        }
        return new Matrix(result);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("max_abs", maxAbs);
        return Collections.unmodifiableMap(params);
    }

    public boolean isFitted() {
        return fitted;
    }
}
