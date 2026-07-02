package org.sklearn.covariance;

import org.sklearn.core.Estimator;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Maximum likelihood covariance estimator.
 *
 * <p>Computes the empirical covariance matrix from training data.
 *
 * <p>Mirrors {@code sklearn.covariance.EmpiricalCovariance}.
 */
public class EmpiricalCovariance implements Estimator<Matrix, Void> {

    private boolean fitted;
    private int nFeatures;
    private Matrix covariance;
    private Matrix precision;
    private Vector mean;
    private boolean assumeCentered;

    public EmpiricalCovariance() {
        this(false);
    }

    public EmpiricalCovariance(boolean assumeCentered) {
        this.assumeCentered = assumeCentered;
    }

    @Override
    public EmpiricalCovariance fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        int m = X.cols();
        nFeatures = m;

        if (!assumeCentered) {
            double[] means = new double[m];
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int i = 0; i < n; i++) sum += X.get(i, j);
                means[j] = sum / n;
            }
            mean = new Vector(means);
        } else {
            mean = new Vector(m);
        }

        double[][] cov = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += (X.get(k, i) - mean.get(i)) * (X.get(k, j) - mean.get(j));
                }
                cov[i][j] = sum / (n - 1);
            }
        }
        covariance = new Matrix(cov);
        precision = invert(cov, m);

        fitted = true;
        return this;
    }

    /**
     * Compute Mahalanobis distances.
     */
    public Vector mahalanobis(Matrix X) {
        Validation.checkFitted(fitted, "EmpiricalCovariance");
        int n = X.rows();
        double[] dist = new double[n];
        for (int i = 0; i < n; i++) {
            double[] diff = new double[nFeatures];
            for (int j = 0; j < nFeatures; j++) {
                diff[j] = X.get(i, j) - mean.get(j);
            }
            double sum = 0;
            for (int a = 0; a < nFeatures; a++) {
                for (int b = 0; b < nFeatures; b++) {
                    sum += diff[a] * precision.get(a, b) * diff[b];
                }
            }
            dist[i] = Math.sqrt(sum);
        }
        return new Vector(dist);
    }

    public Matrix getCovariance() { return covariance; }
    public Matrix getPrecision() {
        if (!fitted) throw new IllegalStateException("EmpiricalCovariance not fitted");
        return precision;
    }
    public Vector getMean() { return mean; }
    public boolean isFitted() { return fitted; }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("covariance_", covariance);
        params.put("precision_", precision);
        params.put("mean_", mean);
        return params;
    }

    private static Matrix invert(double[][] a, int n) {
        double[][] aug = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) aug[i][j] = a[i][j];
            aug[i][n + i] = 1;
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) pivot = row;
            }
            double[] tmp = aug[col]; aug[col] = aug[pivot]; aug[pivot] = tmp;
            double pv = aug[col][col];
            for (int j = 0; j < 2 * n; j++) aug[col][j] /= pv;
            for (int row = 0; row < n; row++) {
                if (row != col) {
                    double f = aug[row][col];
                    for (int j = 0; j < 2 * n; j++) aug[row][j] -= f * aug[col][j];
                }
            }
        }
        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) inv[i][j] = aug[i][n + j];
        }
        return new Matrix(inv);
    }
}
