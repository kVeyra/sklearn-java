package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * PowerTransformer applies a power transformation (Box-Cox or Yeo-Johnson)
 * to make data more Gaussian-like.
 *
 * <p>Mirrors {@code sklearn.preprocessing.PowerTransformer}.
 */
public class PowerTransformer implements Transformer<Matrix, Void> {

    private String method;
    private boolean standardize;
    private boolean fitted;
    private int nFeatures;
    private double[] lambdas;
    private Vector mean;
    private Vector std;

    /**
     * Create PowerTransformer.
     *
     * @param method      "box-cox" or "yeo-johnson"
     * @param standardize whether to zero-mean, unit-variance normalize
     */
    public PowerTransformer(String method, boolean standardize) {
        if (!method.equals("box-cox") && !method.equals("yeo-johnson")) {
            throw new IllegalArgumentException("method must be 'box-cox' or 'yeo-johnson'");
        }
        this.method = method;
        this.standardize = standardize;
    }

    @Override
    public PowerTransformer fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        int m = X.cols();
        nFeatures = m;

        lambdas = new double[m];
        for (int j = 0; j < m; j++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) col[i] = X.get(i, j);
            lambdas[j] = findBestLambda(col);
        }

        if (standardize) {
            double[] means = new double[m];
            double[] stds = new double[m];
            double[][] transformed = new double[n][m];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    transformed[i][j] = applyTransform(X.get(i, j), lambdas[j]);
                }
            }
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int i = 0; i < n; i++) sum += transformed[i][j];
                means[j] = sum / n;
                double var = 0;
                for (int i = 0; i < n; i++) var += Math.pow(transformed[i][j] - means[j], 2);
                stds[j] = Math.sqrt(var / (n - 1));
            }
            mean = new Vector(means);
            std = new Vector(stds);
        }

        fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "PowerTransformer");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Expected " + nFeatures + " features, got " + X.cols());
        }
        int n = X.rows();
        int m = X.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = applyTransform(X.get(i, j), lambdas[j]);
                if (standardize && std.get(j) > 0) {
                    result[i][j] = (result[i][j] - mean.get(j)) / std.get(j);
                }
            }
        }
        return new Matrix(result);
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        Validation.checkFitted(fitted, "PowerTransformer");
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        int m = X.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double val = X.get(i, j);
                if (standardize && std.get(j) > 0) {
                    val = val * std.get(j) + mean.get(j);
                }
                result[i][j] = applyInverse(val, lambdas[j]);
            }
        }
        return new Matrix(result);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("method", method);
        params.put("lambdas_", lambdas);
        return params;
    }

    public double[] getLambdas() { return lambdas; }
    public boolean isFitted() { return fitted; }

    private double applyTransform(double x, double lambda) {
        if (method.equals("box-cox")) {
            if (x <= 0) throw new IllegalArgumentException("Box-Cox requires positive values");
            return Math.abs(lambda) < 1e-10 ? Math.log(x) : (Math.pow(x, lambda) - 1) / lambda;
        } else {
            return yeoJohnsonTransform(x, lambda);
        }
    }

    private double applyInverse(double x, double lambda) {
        if (method.equals("box-cox")) {
            return Math.abs(lambda) < 1e-10 ? Math.exp(x) : Math.pow(x * lambda + 1, 1 / lambda);
        } else {
            return yeoJohnsonInverse(x, lambda);
        }
    }

    private static double yeoJohnsonTransform(double x, double lambda) {
        if (x >= 0) {
            if (Math.abs(lambda) < 1e-10) return Math.log1p(x);
            return (Math.pow(x + 1, lambda) - 1) / lambda;
        } else {
            if (Math.abs(lambda - 2) < 1e-10) return -Math.log1p(-x);
            return -(Math.pow(-x + 1, 2 - lambda) - 1) / (2 - lambda);
        }
    }

    private static double yeoJohnsonInverse(double x, double lambda) {
        if (x >= 0) {
            if (Math.abs(lambda) < 1e-10) return Math.expm1(x);
            return Math.pow(x * lambda + 1, 1 / lambda) - 1;
        } else {
            if (Math.abs(lambda - 2) < 1e-10) return -Math.expm1(-x);
            return -Math.pow((-x) * (2 - lambda) + 1, 1 / (2 - lambda)) + 1;
        }
    }

    private static double findBestLambda(double[] data) {
        int n = data.length;
        double bestLambda = 1.0;
        double bestLogLik = -Double.MAX_VALUE;

        for (double lambda = -2; lambda <= 2; lambda += 0.2) {
            double logLik = computeLogLikelihood(data, lambda);
            if (logLik > bestLogLik) {
                bestLogLik = logLik;
                bestLambda = lambda;
            }
        }
        return bestLambda;
    }

    private static double computeLogLikelihood(double[] data, double lambda) {
        int n = data.length;
        double sum = 0;
        double[] transformed = new double[n];
        for (int i = 0; i < n; i++) {
            if (data[i] <= 0) return -Double.MAX_VALUE;
            transformed[i] = Math.abs(lambda) < 1e-10 ? Math.log(data[i])
                : (Math.pow(data[i], lambda) - 1) / lambda;
            sum += transformed[i];
        }
        double mean = sum / n;
        double var = 0;
        for (int i = 0; i < n; i++) {
            var += (transformed[i] - mean) * (transformed[i] - mean);
        }
        var /= (n - 1);
        if (var <= 0) return -Double.MAX_VALUE;
        double logLik = -n / 2.0 * Math.log(2 * Math.PI) - n / 2.0 * Math.log(var)
            - (n - 1) / 2.0 + (lambda - 1) * sum;
        return logLik;
    }
}
