package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Linear regression with L2 regularization (Tikhonov regularization / Ridge).
 *
 * <p>Minimizes the objective:
 * <pre>
 *   ||X w - y||^2 + alpha * ||w||^2
 * </pre>
 * solved analytically via the normal equations:
 * {@code (X^T X + alpha * I) w = X^T y}.
 *
 * <p>Mirrors {@code sklearn.linear_model.Ridge}.
 *
 * <p>Usage:
 * <pre>{@code
 * Ridge model = new Ridge(1.0);
 * model.fit(X, y);
 * Vector preds = model.predict(X_test);
 * }</pre>
 */
public class Ridge implements Predictor<Matrix, Vector, Vector> {

    private Vector coef;
    private double intercept;
    private double alpha;
    private boolean fitIntercept;
    private boolean fitted;

    /**
     * Create a Ridge model with default alpha=1.0.
     */
    public Ridge() {
        this(1.0, true);
    }

    /**
     * Create a Ridge model.
     *
     * @param alpha regularization strength (must be non-negative)
     */
    public Ridge(double alpha) {
        this(alpha, true);
    }

    /**
     * Create a Ridge model.
     *
     * @param alpha        regularization strength (must be non-negative)
     * @param fitIntercept whether to fit the intercept
     */
    public Ridge(double alpha, boolean fitIntercept) {
        if (alpha < 0) {
            throw new IllegalArgumentException("alpha must be non-negative, got: " + alpha);
        }
        this.alpha = alpha;
        this.fitIntercept = fitIntercept;
    }

    /**
     * Fit the Ridge model to training data.
     *
     * <p>Solves the regularized normal equations
     * {@code (X^T X + alpha * I) w = X^T y} using LU decomposition.
     *
     * @param X training samples, shape (n_samples, n_features)
     * @param y target values, shape (n_samples,)
     * @return this fitted estimator
     */
    @Override
    public Ridge fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();

        Vector yMean = new Vector(1);
        Vector xMean;
        Matrix Xc;
        Vector yc;

        if (fitIntercept) {
            xMean = new Vector(m);
            for (int j = 0; j < m; j++) {
                double sum = 0.0;
                for (int i = 0; i < n; i++) {
                    sum += X.get(i, j);
                }
                xMean.set(j, sum / n);
            }

            double ySum = 0.0;
            for (int i = 0; i < n; i++) {
                ySum += y.get(i);
            }
            yMean.set(0, ySum / n);

            double[][] centered = new double[n][m];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    centered[i][j] = X.get(i, j) - xMean.get(j);
                }
            }
            Xc = new Matrix(centered);

            double[] yCentered = new double[n];
            for (int i = 0; i < n; i++) {
                yCentered[i] = y.get(i) - yMean.get(0);
            }
            yc = new Vector(yCentered);
        } else {
            xMean = Vector.zeros(m);
            Xc = new Matrix(X);
            yc = new Vector(y);
        }

        // Regularized normal equations: (X^T X + alpha * I) w = X^T y
        Matrix XtX = Xc.transpose().multiply(Xc);
        Matrix eye = Matrix.eye(m);
        XtX = XtX.add(eye.multiply(alpha));

        Vector Xty = Xc.transpose().multiply(yc);

        Matrix coefMatrix = XtX.solve(new Matrix(new double[][]{Xty.toArray()}).transpose());
        double[] coefData = new double[m];
        for (int j = 0; j < m; j++) {
            coefData[j] = coefMatrix.get(j, 0);
        }
        this.coef = new Vector(coefData);

        if (fitIntercept) {
            double interceptVal = yMean.get(0);
            for (int j = 0; j < m; j++) {
                interceptVal -= xMean.get(j) * this.coef.get(j);
            }
            this.intercept = interceptVal;
        } else {
            this.intercept = 0.0;
        }

        this.fitted = true;
        return this;
    }

    /**
     * Predict target values for samples in X.
     *
     * @param X samples to predict, shape (n_samples, n_features)
     * @return predicted values, shape (n_samples,)
     */
    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "Ridge");
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
     *
     * @param X test samples, shape (n_samples, n_features)
     * @param y true values, shape (n_samples,)
     * @return R² score
     */
    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "Ridge");
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
        params.put("coef_", coef);
        params.put("intercept_", intercept);
        return Collections.unmodifiableMap(params);
    }
}
