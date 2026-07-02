package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Huber regressor.
 */
public class HuberRegressor implements Predictor<Matrix, Vector, Vector> {

    private double epsilon;
    private int maxIter;
    private double alpha;
    private double tol;
    private boolean fitIntercept;

    private Vector coef;
    private double intercept;
    private double scale;
    private boolean[] outliers;
    private boolean fitted;
    private int nFeatures;

    public HuberRegressor() {
        this(1.35, 100, 0.0001, 1e-5, true);
    }

    public HuberRegressor(double epsilon, int maxIter, double alpha, double tol, boolean fitIntercept) {
        this.epsilon = epsilon;
        this.maxIter = maxIter;
        this.alpha = alpha;
        this.tol = tol;
        this.fitIntercept = fitIntercept;
    }

    @Override
    public HuberRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        double[] w = new double[m];
        double bias = 0.0;
        this.scale = 1.0;

        for (int iter = 0; iter < maxIter; iter++) {
            double[] residuals = new double[n];
            for (int i = 0; i < n; i++) {
                double pred = bias;
                for (int j = 0; j < m; j++) {
                    pred += X.get(i, j) * w[j];
                }
                residuals[i] = y.get(i) - pred;
            }

            // Update scale (robust estimate of standard deviation)
            double median = medianAbsolute(residuals);
            this.scale = Math.max(epsilon * 1e-10, median / 0.6745);

            double[] grad = new double[m];
            double gradBias = 0.0;
            double loss = 0.0;

            for (int i = 0; i < n; i++) {
                double r = residuals[i];
                double z = r / scale;
                double absZ = Math.abs(z);
                double wI;
                double wIDeriv;

                if (absZ <= epsilon) {
                    wI = 0.5 * z * z;
                    wIDeriv = z;
                } else {
                    wI = epsilon * absZ - 0.5 * epsilon * epsilon;
                    wIDeriv = epsilon * Math.signum(z);
                }

                loss += wI;
                double contrib = wIDeriv / scale;
                for (int j = 0; j < m; j++) {
                    grad[j] += contrib * X.get(i, j);
                }
                gradBias += contrib;
            }

            // Add L2 regularization (not on intercept)
            for (int j = 0; j < m; j++) {
                loss += 0.5 * alpha * w[j] * w[j];
                grad[j] += alpha * w[j];
            }

            // Gradient descent step with line search
            double step = 1.0;
            for (int ls = 0; ls < 30; ls++) {
                double[] wNew = new double[m];
                for (int j = 0; j < m; j++) {
                    wNew[j] = w[j] - step * grad[j];
                }
                double biasNew = bias - step * gradBias;

                double newLoss = 0.0;
                for (int i = 0; i < n; i++) {
                    double pred = biasNew;
                    for (int j = 0; j < m; j++) {
                        pred += X.get(i, j) * wNew[j];
                    }
                    double r = y.get(i) - pred;
                    double z = r / scale;
                    double absZ = Math.abs(z);
                    newLoss += (absZ <= epsilon) ? 0.5 * z * z : epsilon * absZ - 0.5 * epsilon * epsilon;
                }
                for (int j = 0; j < m; j++) {
                    newLoss += 0.5 * alpha * wNew[j] * wNew[j];
                }

                if (newLoss <= loss + 1e-4 * step * dotProduct(grad, grad, gradBias, gradBias)) {
                    w = wNew;
                    bias = biasNew;
                    break;
                }
                step *= 0.5;
            }

            // Check convergence
            double wChange = 0.0;
            for (int j = 0; j < m; j++) {
                wChange += Math.abs(grad[j]);
            }
            if (wChange < tol) {
                break;
            }
        }

        this.coef = new Vector(w);
        this.intercept = bias;

        // Identify outliers
        this.outliers = new boolean[n];
        for (int i = 0; i < n; i++) {
            double pred = bias;
            for (int j = 0; j < m; j++) {
                pred += X.get(i, j) * w[j];
            }
            double res = Math.abs(y.get(i) - pred);
            outliers[i] = res > scale * epsilon;
        }

        this.fitted = true;
        return this;
    }

    private double medianAbsolute(double[] arr) {
        double[] absSorted = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            absSorted[i] = Math.abs(arr[i]);
        }
        Arrays.sort(absSorted);
        int n = arr.length;
        if (n % 2 == 0) {
            return (absSorted[n / 2 - 1] + absSorted[n / 2]) / 2.0;
        }
        return absSorted[n / 2];
    }

    private double dotProduct(double[] a, double[] b, double ba, double bb) {
        double sum = ba * bb;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return -sum;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "HuberRegressor");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Feature dimension mismatch: expected " + nFeatures + " got " + X.cols());
        }

        int n = X.rows();
        double[] preds = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = intercept;
            for (int j = 0; j < nFeatures; j++) {
                sum += X.get(i, j) * coef.get(j);
            }
            preds[i] = sum;
        }
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "HuberRegressor");
        Vector pred = predict(X);
        double ssRes = 0.0, ssTot = 0.0;
        double yMean = y.mean();
        for (int i = 0; i < y.size(); i++) {
            double diff = y.get(i) - pred.get(i);
            ssRes += diff * diff;
            double diffMean = y.get(i) - yMean;
            ssTot += diffMean * diffMean;
        }
        if (ssTot == 0.0) {
            return 1.0;
        }
        return 1.0 - ssRes / ssTot;
    }

    public Vector getCoef() {
        return coef;
    }
    public double getIntercept() {
        return intercept;
    }
    public double getScale() {
        return scale;
    }
    public boolean[] getOutliers() {
        return outliers;
    }
    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("epsilon", epsilon); p.put("max_iter", maxIter);
        p.put("alpha", alpha); p.put("tol", tol);
        p.put("fit_intercept", fitIntercept);
        return Collections.unmodifiableMap(p);
    }
}
