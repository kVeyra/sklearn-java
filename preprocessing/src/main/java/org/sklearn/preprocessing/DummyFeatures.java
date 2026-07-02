package org.sklearn.preprocessing;

import org.sklearn.math.Matrix;

/**
 * Utilities for generating dummy features.
 *
 * <p>Mirrors {@code sklearn.preprocessing.add_dummy_feature}.
 */
public final class DummyFeatures {

    private DummyFeatures() {
    }

    /**
     * Augment dataset with an intercept column (all ones).
     *
     * @param X       input data
     * @param value   value to fill the dummy feature column
     * @return augmented matrix with an extra column of ones at the beginning
     */
    public static Matrix addDummyFeature(Matrix X, double value) {
        int n = X.rows();
        int m = X.cols();
        double[][] data = new double[n][m + 1];
        for (int i = 0; i < n; i++) {
            data[i][0] = value;
            for (int j = 0; j < m; j++) {
                data[i][j + 1] = X.get(i, j);
            }
        }
        return new Matrix(data);
    }
}
