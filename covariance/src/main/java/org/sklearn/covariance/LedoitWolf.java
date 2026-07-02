package org.sklearn.covariance;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Ledoit-Wolf optimal shrinkage covariance estimator.
 *
 * <p>Computes a shrinkage covariance estimate with an analytically
 * determined shrinkage coefficient.
 *
 * <p>Mirrors {@code sklearn.covariance.LedoitWolf}.
 */
public class LedoitWolf extends EmpiricalCovariance {

    private double shrinkage;
    private Matrix shrunkCovariance;

    public LedoitWolf() {
        super(false);
    }

    public LedoitWolf(boolean assumeCentered) {
        super(assumeCentered);
    }

    @Override
    public LedoitWolf fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        int m = X.cols();

        super.fit(X, y);
        Matrix empCov = getCovariance();
        Vector mu = getMean();

        double[][] s = empCov.toArray();

        double[] var = new double[m * m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    double xi = X.get(k, i) - mu.get(i);
                    double xj = X.get(k, j) - mu.get(j);
                    double diff = xi * xj - s[i][j];
                    sum += diff * diff;
                }
                var[i * m + j] = sum / (n - 1);
            }
        }

        double sumVar = 0;
        for (double v : var) sumVar += v;

        double sumSq = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                sumSq += s[i][j] * s[i][j];
            }
        }

        double traceSq = 0;
        for (int i = 0; i < m; i++) {
            traceSq += s[i][i] * s[i][i];
        }

        double num = sumVar;
        double denom = sumSq - traceSq / m;

        shrinkage = Math.max(0, Math.min(1, num / Math.max(denom, 1e-15)));

        double[][] shrunk = new double[m][m];
        double target = traceSq / m;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                if (i == j) {
                    shrunk[i][j] = (1 - shrinkage) * s[i][j] + shrinkage * target;
                } else {
                    shrunk[i][j] = (1 - shrinkage) * s[i][j];
                }
            }
        }
        shrunkCovariance = new Matrix(shrunk);

        return this;
    }

    public double getShrinkage() { return shrinkage; }
    public Matrix getShrunkCovariance() { return shrunkCovariance; }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("shrinkage", shrinkage);
        params.put("shrunk_covariance_", shrunkCovariance);
        params.put("covariance_", getCovariance());
        return params;
    }
}
