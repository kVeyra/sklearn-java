package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Linear regression with combined L1 and L2 regularization (ElasticNet).
 *
 * <p>Minimizes the objective:
 * <pre>
 *   1/(2*n) * ||X w - y||^2
 *     + alpha * l1_ratio * ||w||_1
 *     + alpha * (1 - l1_ratio) / 2 * ||w||^2
 * </pre>
 * solved via coordinate descent with soft-thresholding.
 *
 * <p>L1 regularization encourages sparsity; L2 regularization handles
 * correlated features and stabilizes the solution.
 *
 * <p>Mirrors {@code sklearn.linear_model.ElasticNet}.
 *
 * <p>Usage:
 * <pre>{@code
 * ElasticNet model = new ElasticNet(0.1, 0.5);
 * model.fit(X, y);
 * Vector preds = model.predict(X_test);
 * }</pre>
 */
public class ElasticNet implements Predictor<Matrix, Vector, Vector> {

    private Vector coef;
    private double intercept;
    private double alpha;
    private double l1Ratio;
    private double tol;
    private int maxIter;
    private boolean fitIntercept;
    private boolean fitted;

    /**
     * Create an ElasticNet model with default alpha=1.0 and l1_ratio=0.5.
     */
    public ElasticNet() {
        this(1.0, 0.5);
    }

    /**
     * Create an ElasticNet model.
     *
     * @param alpha   regularization strength (must be non-negative)
     * @param l1Ratio mixing parameter between 0 and 1 (0 = Ridge, 1 = Lasso)
     */
    public ElasticNet(double alpha, double l1Ratio) {
        this(alpha, l1Ratio, true, 1e-4, 1000);
    }

    /**
     * Create an ElasticNet model.
     *
     * @param alpha        regularization strength (must be non-negative)
     * @param l1Ratio      mixing parameter between 0 and 1
     * @param fitIntercept whether to fit the intercept
     * @param tol          tolerance for stopping criterion
     * @param maxIter      maximum number of iterations
     */
    public ElasticNet(double alpha, double l1Ratio, boolean fitIntercept,
                      double tol, int maxIter) {
        if (alpha < 0) {
            throw new IllegalArgumentException("alpha must be non-negative, got: " + alpha);
        }
        if (l1Ratio < 0 || l1Ratio > 1) {
            throw new IllegalArgumentException(
                "l1_ratio must be between 0 and 1, got: " + l1Ratio);
        }
        this.alpha = alpha;
        this.l1Ratio = l1Ratio;
        this.fitIntercept = fitIntercept;
        this.tol = tol;
        this.maxIter = maxIter;
    }

    /**
     * Fit the ElasticNet model using coordinate descent.
     *
     * @param X training samples, shape (n_samples, n_features)
     * @param y target values, shape (n_samples,)
     * @return this fitted estimator
     */
    @Override
    public ElasticNet fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();

        double[] w = new double[m];
        double b = 0.0;

        // Precompute column norms (sum of squares)
        double[] norms = new double[m];
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                double val = X.get(i, j);
                sum += val * val;
            }
            norms[j] = sum;
        }

        double l1Penalty = n * alpha * l1Ratio;
        double l2Penalty = n * alpha * (1.0 - l1Ratio);

        // Initialize residuals
        double[] residuals = new double[n];
        System.arraycopy(y.toArray(), 0, residuals, 0, n);

        for (int iter = 0; iter < maxIter; iter++) {
            double maxChange = 0.0;

            for (int j = 0; j < m; j++) {
                if (norms[j] < 1e-15) {
                    continue;
                }

                double rho = 0.0;
                for (int i = 0; i < n; i++) {
                    rho += X.get(i, j) * residuals[i];
                }
                rho += w[j] * norms[j];
                double oldWj = w[j];
                double denominator = norms[j] + l2Penalty;

                // Soft-thresholding update with L2 penalty in denominator
                double wNew;
                if (rho > l1Penalty) {
                    wNew = (rho - l1Penalty) / denominator;
                } else if (rho < -l1Penalty) {
                    wNew = (rho + l1Penalty) / denominator;
                } else {
                    wNew = 0.0;
                }

                w[j] = wNew;
                double change = Math.abs(wNew - oldWj);
                if (change > maxChange) {
                    maxChange = change;
                }

                // Update residuals
                double delta = wNew - oldWj;
                if (delta != 0.0) {
                    for (int i = 0; i < n; i++) {
                        residuals[i] -= delta * X.get(i, j);
                    }
                }
            }

            // Update intercept
            if (fitIntercept) {
                double newB = 0.0;
                for (int i = 0; i < n; i++) {
                    newB += y.get(i);
                    for (int j = 0; j < m; j++) {
                        newB -= X.get(i, j) * w[j];
                    }
                }
                newB /= n;
                double bChange = Math.abs(newB - b);
                b = newB;
                if (bChange > maxChange) {
                    maxChange = bChange;
                }

                // Recompute residuals
                for (int i = 0; i < n; i++) {
                    double pred = b;
                    for (int j = 0; j < m; j++) {
                        pred += X.get(i, j) * w[j];
                    }
                    residuals[i] = y.get(i) - pred;
                }
            }

            if (maxChange < tol) {
                break;
            }
        }

        this.coef = new Vector(w);
        this.intercept = b;
        this.fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "ElasticNet");
        Validation.checkMatrix(X, -1);
        if (X.cols() != coef.size()) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + coef.size()
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        double[] preds = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = intercept;
            for (int j = 0; j < coef.size(); j++) {
                sum += X.get(i, j) * coef.get(j);
            }
            preds[i] = sum;
        }
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "ElasticNet");
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        Vector pred = predict(X);
        double ssRes = 0.0;
        double ssTot = 0.0;
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

    public double getAlpha() {
        return alpha;
    }

    public double getL1Ratio() {
        return l1Ratio;
    }

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("alpha", alpha);
        params.put("l1_ratio", l1Ratio);
        params.put("fit_intercept", fitIntercept);
        params.put("tol", tol);
        params.put("max_iter", maxIter);
        params.put("coef_", coef);
        params.put("intercept_", intercept);
        return Collections.unmodifiableMap(params);
    }
}
