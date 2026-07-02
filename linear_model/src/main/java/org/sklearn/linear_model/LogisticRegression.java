package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Logistic Regression (aka logit, MaxEnt) classifier.
 *
 * <p>Implements regularized logistic regression using gradient descent
 * with backtracking line search. Supports binary classification with
 * L2 regularization.
 *
 * <p>The objective function minimized is:
 * <pre>
 *   J(w) = sum_i loss(y_i, x_i^T w + b) + 1/(2*C) * ||w||^2
 * </pre>
 * where {@code loss(y, z) = -y*log(p) - (1-y)*log(1-p)} and
 * {@code p = 1/(1 + exp(-z))}.
 *
 * <p>Mirrors {@code sklearn.linear_model.LogisticRegression} with solver='lbfgs'
 * for binary classification.
 *
 * <p>Usage:
 * <pre>{@code
 * LogisticRegression model = new LogisticRegression();
 * model.fit(X, y);
 * Vector preds = model.predict(X_test);
 * Vector probs = model.predictProba(X_test);
 * double acc = model.score(X_test, y_test);
 * }</pre>
 */
public class LogisticRegression implements Predictor<Matrix, Vector, Vector> {

    private Vector coef;
    private Vector intercept;
    private double regularizationC;
    private double tol;
    private int maxIter;
    private boolean fitIntercept;
    private boolean fitted;
    private int[] classes;
    private int nFeatures;

    /**
     * Create a logistic regression classifier with default parameters.
     */
    public LogisticRegression() {
        this(1.0, true, 1e-4, 100);
    }

    /**
     * Create a logistic regression classifier.
     *
     * @param C            inverse of regularization strength (must be positive)
     * @param fitIntercept whether to fit the intercept
     * @param tol          tolerance for stopping criterion
     * @param maxIter      maximum number of iterations
     */
    public LogisticRegression(double C, boolean fitIntercept, double tol, int maxIter) {
        if (C <= 0) {
            throw new IllegalArgumentException("C must be positive, got: " + C);
        }
        if (tol <= 0) {
            throw new IllegalArgumentException("tol must be positive, got: " + tol);
        }
        if (maxIter <= 0) {
            throw new IllegalArgumentException("maxIter must be positive, got: " + maxIter);
        }
        this.regularizationC = C;
        this.fitIntercept = fitIntercept;
        this.tol = tol;
        this.maxIter = maxIter;
    }

    /**
     * Fit the logistic regression model to training data.
     *
     * @param X training samples, shape (n_samples, n_features)
     * @param y target values, shape (n_samples,)
     * @return this fitted estimator
     */
    @Override
    public LogisticRegression fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        // Encode class labels
        Set<Integer> uniqueLabels = new LinkedHashSet<>();
        for (int i = 0; i < y.size(); i++) {
            uniqueLabels.add((int) y.get(i));
        }
        this.classes = uniqueLabels.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(this.classes);

        int nClasses = classes.length;
        if (nClasses < 2) {
            throw new IllegalArgumentException(
                "LogisticRegression requires at least 2 classes, got " + nClasses);
        }

        if (nClasses > 2) {
            // For multiclass, we could implement OvR or multinomial.
            // For now, raise a clear error about binary-only support.
            throw new UnsupportedOperationException(
                "Multiclass classification is not yet supported. "
                    + "Found " + nClasses + " classes. "
                    + "Use OneVsRestClassifier or upgrade to a future version.");
        }

        // Binary classification: encode positive class as 1, negative as 0
        int posLabel = classes[1];
        double[] yBinary = new double[n];
        for (int i = 0; i < n; i++) {
            yBinary[i] = (y.get(i) == posLabel) ? 1.0 : 0.0;
        }
        Vector yBin = new Vector(yBinary);

        // Initialize weights
        int nParams = fitIntercept ? m + 1 : m;
        double[] w = new double[nParams];

        // Gradient descent with backtracking line search
        double[] grad = new double[nParams];
        double[] dir = new double[nParams];

        double l2Reg = 1.0 / regularizationC;
        double invN = 1.0 / n;

        for (int iter = 0; iter < maxIter; iter++) {
            // Compute loss and gradient
            double loss = 0.0;
            for (int j = 0; j < nParams; j++) {
                grad[j] = 0.0;
            }

            for (int i = 0; i < n; i++) {
                double z = fitIntercept ? w[m] : 0.0;
                for (int j = 0; j < m; j++) {
                    z += X.get(i, j) * w[j];
                }
                double p = sigmoid(z);
                double yi = yBin.get(i);
                loss += -yi * logSafe(p) - (1.0 - yi) * logSafe(1.0 - p);

                double diff = p - yi;
                for (int j = 0; j < m; j++) {
                    grad[j] += diff * X.get(i, j);
                }
                if (fitIntercept) {
                    grad[m] += diff;
                }
            }

            // Add regularization (not on intercept)
            double regLoss = 0.0;
            for (int j = 0; j < m; j++) {
                regLoss += 0.5 * l2Reg * w[j] * w[j];
                grad[j] += l2Reg * w[j];
            }
            loss += regLoss;

            // Check convergence (infinity norm of gradient)
            double gradNorm = 0.0;
            for (int j = 0; j < nParams; j++) {
                gradNorm = Math.max(gradNorm, Math.abs(grad[j]));
            }
            if (gradNorm < tol) {
                break;
            }

            // Compute search direction (steepest descent)
            for (int j = 0; j < nParams; j++) {
                dir[j] = -grad[j];
            }

            // Backtracking line search (Armijo condition)
            double step = 1.0;
            double c = 1e-4;
            double dotGradDir = 0.0;
            for (int j = 0; j < nParams; j++) {
                dotGradDir += grad[j] * dir[j];
            }

            for (int ls = 0; ls < 50; ls++) {
                double[] wNew = new double[nParams];
                for (int j = 0; j < nParams; j++) {
                    wNew[j] = w[j] + step * dir[j];
                }

                // Compute loss at new point
                double newLoss = 0.0;
                for (int i = 0; i < n; i++) {
                    double z = fitIntercept ? wNew[m] : 0.0;
                    for (int j = 0; j < m; j++) {
                        z += X.get(i, j) * wNew[j];
                    }
                    double p = sigmoid(z);
                    double yi = yBin.get(i);
                    newLoss += -yi * logSafe(p) - (1.0 - yi) * logSafe(1.0 - p);
                }
                for (int j = 0; j < m; j++) {
                    newLoss += 0.5 * l2Reg * wNew[j] * wNew[j];
                }

                if (newLoss <= loss + c * step * dotGradDir) {
                    w = wNew;
                    break;
                }
                step *= 0.5;
            }
        }

        // Extract coefficients and intercept
        double[] coefData = new double[m];
        System.arraycopy(w, 0, coefData, 0, m);
        this.coef = new Vector(coefData);

        if (fitIntercept) {
            this.intercept = new Vector(new double[]{w[m]});
        } else {
            this.intercept = new Vector(new double[]{0.0});
        }

        this.fitted = true;
        return this;
    }

    /**
     * Predict class labels for samples in X.
     *
     * @param X samples to predict, shape (n_samples, n_features)
     * @return predicted class labels, shape (n_samples,)
     */
    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "LogisticRegression");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }

        Vector probs = predictProba(X);
        int n = X.rows();
        double[] preds = new double[n];
        int posLabel = classes[1];
        int negLabel = classes[0];
        for (int i = 0; i < n; i++) {
            preds[i] = probs.get(i) >= 0.5 ? posLabel : negLabel;
        }
        return new Vector(preds);
    }

    /**
     * Predict class probabilities for samples in X.
     *
     * @param X samples to predict, shape (n_samples, n_features)
     * @return probability of the positive class, shape (n_samples,)
     */
    public Vector predictProba(Matrix X) {
        Validation.checkFitted(fitted, "LogisticRegression");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        int m = X.cols();
        double[] probs = new double[n];
        for (int i = 0; i < n; i++) {
            double z = intercept.get(0);
            for (int j = 0; j < m; j++) {
                z += X.get(i, j) * coef.get(j);
            }
            probs[i] = sigmoid(z);
        }
        return new Vector(probs);
    }

    /**
     * Compute accuracy score.
     *
     * @param X test samples, shape (n_samples, n_features)
     * @param y true labels, shape (n_samples,)
     * @return accuracy score (fraction of correct predictions)
     */
    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "LogisticRegression");
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        Vector pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y.size(); i++) {
            if (Math.abs(pred.get(i) - y.get(i)) < 0.5) {
                correct++;
            }
        }
        return (double) correct / y.size();
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
     * @return intercept value, shape (1,)
     */
    public Vector getIntercept() {
        return intercept;
    }

    /**
     * Return the class labels known to the classifier.
     */
    public int[] getClasses() {
        return classes;
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
        params.put("C", regularizationC);
        params.put("fit_intercept", fitIntercept);
        params.put("tol", tol);
        params.put("max_iter", maxIter);
        params.put("coef_", coef);
        params.put("intercept_", intercept);
        params.put("classes_", classes);
        return Collections.unmodifiableMap(params);
    }

    private static double sigmoid(double z) {
        if (z > 40) {
            return 1.0;
        }
        if (z < -40) {
            return 0.0;
        }
        return 1.0 / (1.0 + Math.exp(-z));
    }

    private static double logSafe(double x) {
        if (x <= 0.0) {
            return 0.0;
        }
        if (x >= 1.0) {
            return 0.0;
        }
        return Math.log(x);
    }
}
