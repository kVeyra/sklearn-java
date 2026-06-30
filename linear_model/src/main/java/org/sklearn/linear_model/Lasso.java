package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Linear regression with L1 regularization (Lasso).
 *
 * <p>Minimizes the objective:
 * <pre>
 *   1/(2*n) * ||X w - y||^2 + alpha * ||w||_1
 * </pre>
 * solved via coordinate descent with soft-thresholding.
 *
 * <p>Mirrors {@code sklearn.linear_model.Lasso}.
 *
 * <p>Usage:
 * <pre>{@code
 * Lasso model = new Lasso(0.1);
 * model.fit(X, y);
 * Vector preds = model.predict(X_test);
 * }</pre>
 */
public class Lasso implements Predictor<Matrix, Vector, Vector> {

    private Vector coef;
    private double intercept;
    private double alpha;
    private double tol;
    private int maxIter;
    private boolean fitIntercept;
    private boolean fitted;

    /**
     * Create a Lasso model with default alpha=1.0.
     */
    public Lasso() {
        this(1.0);
    }

    /**
     * Create a Lasso model.
     *
     * @param alpha regularization strength (must be non-negative)
     */
    public Lasso(double alpha) {
        this(alpha, true, 1e-4, 1000);
    }

    /**
     * Create a Lasso model.
     *
     * @param alpha        regularization strength (must be non-negative)
     * @param fitIntercept whether to fit the intercept
     * @param tol          tolerance for stopping criterion
     * @param maxIter      maximum number of iterations
     */
    public Lasso(double alpha, boolean fitIntercept, double tol, int maxIter) {
        if (alpha < 0) {
            throw new IllegalArgumentException("alpha must be non-negative, got: " + alpha);
        }
        this.alpha = alpha;
        this.fitIntercept = fitIntercept;
        this.tol = tol;
        this.maxIter = maxIter;
    }

    /**
     * Fit the Lasso model using coordinate descent.
     *
     * @param X training samples, shape (n_samples, n_features)
     * @param y target values, shape (n_samples,)
     * @return this fitted estimator
     */
    @Override
    public Lasso fit(Matrix X, Vector y) {
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

        // Compute residuals
        double[] residuals = new double[n];
        System.arraycopy(y.toArray(), 0, residuals, 0, n);

        double nAlpha = n * alpha;

        for (int iter = 0; iter < maxIter; iter++) {
            double maxChange = 0.0;

            for (int j = 0; j < m; j++) {
                if (norms[j] < 1e-15) {
                    continue;
                }

                // Compute rho_j = sum_i X_{i,j} * (residual_i + w_j * X_{i,j})
                // where residual_i already includes w_j's contribution, so we add
                // it back to get the partial residual excluding feature j.
                double rho = 0.0;
                for (int i = 0; i < n; i++) {
                    rho += X.get(i, j) * residuals[i];
                }
                rho += w[j] * norms[j];
                double oldWj = w[j];

                // Soft-thresholding update
                double wNew;
                if (rho > nAlpha) {
                    wNew = (rho - nAlpha) / norms[j];
                } else if (rho < -nAlpha) {
                    wNew = (rho + nAlpha) / norms[j];
                } else {
                    wNew = 0.0;
                }

                w[j] = wNew;
                double change = Math.abs(wNew - oldWj);
                if (change > maxChange) {
                    maxChange = change;
                }

                // Update residuals: subtract (wNew - oldWj) * X_{:,j}
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

                // Recompute residuals with new intercept
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

    /**
     * Predict target values for samples in X.
     */
    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "Lasso");
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

    /**
     * Compute R² score.
     */
    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "Lasso");
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

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("alpha", alpha);
        params.put("fit_intercept", fitIntercept);
        params.put("tol", tol);
        params.put("max_iter", maxIter);
        params.put("coef_", coef);
        params.put("intercept_", intercept);
        return Collections.unmodifiableMap(params);
    }
}
