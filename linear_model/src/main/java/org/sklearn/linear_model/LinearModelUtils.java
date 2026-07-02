package org.sklearn.linear_model;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

/**
 * Static utility functions for linear models.
 *
 * <p>Mirrors {@code sklearn.linear_model.ridge_regression} and related functions.
 */
public final class LinearModelUtils {

    private LinearModelUtils() {
    }

    /**
     * Solve ridge regression: (X^T X + alpha*I) w = X^T y
     *
     * @param X     feature matrix
     * @param y     target vector
     * @param alpha regularization strength
     * @return coefficient vector
     */
    public static Vector ridgeRegression(Matrix X, Vector y, double alpha) {
        int n = X.rows();
        int m = X.cols();

        double[][] xtx = new double[m][m];
        double[] xty = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += X.get(k, i) * X.get(k, j);
                }
                xtx[i][j] = sum;
            }
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += X.get(k, i) * y.get(k);
            }
            xty[i] = sum;
        }

        for (int i = 0; i < m; i++) {
            xtx[i][i] += alpha;
        }

        double[] coef = solveLinear(xtx, xty, m);
        return new Vector(coef);
    }

    private static double[] solveLinear(double[][] a, double[] b, int n) {
        double[][] aug = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, aug[i], 0, n);
            aug[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) {
                    pivot = row;
                }
            }
            double[] tmp = aug[col];
            aug[col] = aug[pivot];
            aug[pivot] = tmp;
            double pivotVal = aug[col][col];
            if (Math.abs(pivotVal) < 1e-15) continue;
            for (int j = col; j <= n; j++) {
                aug[col][j] /= pivotVal;
            }
            for (int row = 0; row < n; row++) {
                if (row != col) {
                    double factor = aug[row][col];
                    for (int j = col; j <= n; j++) {
                        aug[row][j] -= factor * aug[col][j];
                    }
                }
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = aug[i][n];
        }
        return x;
    }
}
