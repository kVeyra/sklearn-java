package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Ordinary least squares Linear Regression.
 *
 * <p>Fits a linear model {@code y = X * coef_ + intercept_} by minimizing
 * the residual sum of squares between observed and predicted targets.
 * The coefficients are computed via the normal equations:
 * {@code (X^T X) coef = X^T y}, solved using LU decomposition with partial
 * pivoting.
 *
 * <p>Mirrors {@code sklearn.linear_model.LinearRegression}. For well-conditioned
 * problems the results match sklearn's SVD-based solver to within 1e-10.
 *
 * <p>Usage:
 * <pre>{@code
 * LinearRegression model = new LinearRegression();
 * model.fit(X, y);
 * Vector predictions = model.predict(X_test);
 * double r2 = model.score(X_test, y_test);
 * }</pre>
 */
public class LinearRegression implements Predictor<Matrix, Vector, Vector> {

    private Vector coef;
    private double intercept;
    private boolean fitIntercept;
    private boolean fitted;

    /**
     * Create a linear regression model with intercept fitting enabled.
     */
    public LinearRegression() {
        this(true);
    }

    /**
     * Create a linear regression model.
     *
     * @param fitIntercept whether to fit the intercept (bias) term
     */
    public LinearRegression(boolean fitIntercept) {
        this.fitIntercept = fitIntercept;
    }

    /**
     * Fit the linear model to training data.
     *
     * <p>Solves the normal equations {@code (X^T X) w = X^T y} using
     * LU decomposition. When {@code fitIntercept} is true, the data
     * is centered before solving and the intercept is computed as
     * {@code y_mean - X_mean @ coef}.
     *
     * @param X training samples, shape (n_samples, n_features)
     * @param y target values, shape (n_samples,)
     * @return this fitted estimator
     */
    @Override
    public LinearRegression fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        int p = fitIntercept ? m : m;

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

        // Normal equations: (X^T X) coef = X^T y
        Matrix XtX = Xc.transpose().multiply(Xc);
        Vector Xty = Xc.transpose().multiply(yc);

        // For underdetermined systems (n < p), regularize slightly
        if (n <= p) {
            Matrix eye = Matrix.eye(p);
            XtX = XtX.add(eye.multiply(1e-8));
        }

        try {
            Matrix coefMatrix = XtX.solve(new Matrix(new double[][]{Xty.toArray()}).transpose());
            double[] coefData = new double[p];
            for (int j = 0; j < p; j++) {
                coefData[j] = coefMatrix.get(j, 0);
            }
            this.coef = new Vector(coefData);
        } catch (IllegalStateException e) {
            // Fallback: use pseudo-inverse via regularization
            Matrix eye = Matrix.eye(p);
            XtX = Xc.transpose().multiply(Xc).add(eye.multiply(1e-6));
            Matrix coefMatrix = XtX.solve(new Matrix(new double[][]{Xty.toArray()}).transpose());
            double[] coefData = new double[p];
            for (int j = 0; j < p; j++) {
                coefData[j] = coefMatrix.get(j, 0);
            }
            this.coef = new Vector(coefData);
        }

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
        Validation.checkFitted(fitted, "LinearRegression");
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
     * Compute the coefficient of determination R².
     *
     * @param X test samples, shape (n_samples, n_features)
     * @param y true values, shape (n_samples,)
     * @return R² score (1.0 is perfect prediction)
     */
    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "LinearRegression");
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

    /**
     * Return the fitted coefficients.
     *
     * @return coefficient vector, shape (n_features,)
     */
    public Vector getCoef() {
        return coef;
    }

    /**
     * Return the fitted intercept.
     *
     * @return intercept value
     */
    public double getIntercept() {
        return intercept;
    }

    /**
     * Check whether this model has been fitted.
     */
    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("fit_intercept", fitIntercept);
        params.put("coef_", coef);
        params.put("intercept_", intercept);
        return Collections.unmodifiableMap(params);
    }
}
