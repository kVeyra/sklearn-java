package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Bayesian ridge regression.
 */
public class BayesianRidge implements Predictor<Matrix, Vector, Vector> {

    private int maxIter;
    private double tol;
    private double alpha1;
    private double alpha2;
    private double lambda1;
    private double lambda2;
    private boolean fitIntercept;
    private boolean computeScore;

    private Vector coef;
    private double intercept;
    private double alpha;
    private double lambda;
    private Matrix sigma;
    private double[] scores;
    private int nIter;
    private boolean fitted;
    private int nFeatures;

    public BayesianRidge() {
        this(300, 1e-3, 1e-6, 1e-6, 1e-6, 1e-6, true, false);
    }

    public BayesianRidge(int maxIter, double tol, double alpha1, double alpha2,
                          double lambda1, double lambda2, boolean fitIntercept,
                          boolean computeScore) {
        this.maxIter = maxIter;
        this.tol = tol;
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
        this.fitIntercept = fitIntercept;
        this.computeScore = computeScore;
    }

    @Override
    public BayesianRidge fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        // Center data
        double[] xMean = new double[m];
        if (fitIntercept) {
            for (int j = 0; j < m; j++) {
                double sum = 0.0;
                for (int i = 0; i < n; i++) {
                    sum += X.get(i, j);
                }
                xMean[j] = sum / n;
            }
        }
        double yMean = fitIntercept ? y.mean() : 0.0;

        double[][] XcArr = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                XcArr[i][j] = X.get(i, j) - xMean[j];
            }
        }
        Matrix Xc = new Matrix(XcArr);

        double[] ycArr = new double[n];
        for (int i = 0; i < n; i++) {
            ycArr[i] = y.get(i) - yMean;
        }
        Vector yc = new Vector(ycArr);

        // Compute X^T X and X^T y
        Matrix XtX = Xc.transpose().multiply(Xc);
        Vector Xty = Xc.transpose().multiply(yc);

        // Initialize alpha and lambda
        double yVar = 0.0;
        for (int i = 0; i < n; i++) {
            yVar += ycArr[i] * ycArr[i];
        }
        yVar = Math.max(yVar / n, 1e-10);
        this.alpha = 1.0 / yVar;
        this.lambda = 1.0;

        double[][] scoresList = computeScore ? new double[maxIter][2] : null;

        // EM-style iterative update
        for (int iter = 0; iter < maxIter; iter++) {
            Matrix A = XtX.multiply(alpha).add(Matrix.eye(m).multiply(lambda));

            Vector w;
            try {
                Matrix wMat = A.solve(new Matrix(new double[][]{Xty.toArray()}).transpose());
                double[] wData = new double[m];
                for (int j = 0; j < m; j++) {
                    wData[j] = wMat.get(j, 0);
                }
                w = new Vector(wData);
            } catch (Exception e) {
                break;
            }

            // Compute SSE
            double sse = 0.0;
            for (int i = 0; i < n; i++) {
                double pred = 0.0;
                for (int j = 0; j < m; j++) {
                    pred += XcArr[i][j] * w.get(j);
                }
                double diff = ycArr[i] - pred;
                sse += diff * diff;
            }

            // Compute gamma = sum of alpha * eigen_i / (lambda + alpha * eigen_i)
            // Approximate using trace of (A^-1 * X^T X)
            double gamma = 0.0;
            try {
                Matrix Ainv = A.inverse();
                for (int j = 0; j < m; j++) {
                    for (int k = 0; k < m; k++) {
                        gamma += alpha * Ainv.get(j, k) * XtX.get(k, j);
                    }
                }
            } catch (Exception e) {
                gamma = m;
            }

            // Update lambda and alpha
            double wSq = 0.0;
            for (int j = 0; j < m; j++) {
                wSq += w.get(j) * w.get(j);
            }

            this.lambda = Math.max((gamma + 2 * lambda1) / (Math.max(wSq, 1e-10) + 2 * lambda2), 1e-10);
            this.alpha = Math.max((n - gamma + 2 * alpha1) / (Math.max(sse, 1e-10) + 2 * alpha2), 1e-10);

            // Check for divergence
            boolean diverged = false;
            for (int j = 0; j < m; j++) {
                if (Double.isNaN(w.get(j)) || Double.isInfinite(w.get(j))) {
                    diverged = true;
                    break;
                }
            }
            if (diverged) {
                break;
            }

            if (computeScore && scoresList != null) {
                scoresList[iter][0] = alpha;
                scoresList[iter][1] = lambda;
            }

            // Keep w for next iteration but only use on convergence
            if (iter > 0) {
                double change = 0.0;
                for (int j = 0; j < m; j++) {
                    change += Math.abs(w.get(j) - coef.get(j));
                }
                if (change < tol && iter > 3) {
                    this.coef = w;
                    this.nIter = iter + 1;
                    // Compute posterior sigma
                    computePosteriorSigma(A);
                    this.fitted = true;
                    return this;
                }
            }
            this.coef = w;
        }

        // Compute posterior sigma
        Matrix A = XtX.multiply(alpha).add(Matrix.eye(m).multiply(lambda));
        computePosteriorSigma(A);

        if (fitIntercept) {
            this.intercept = yMean;
            for (int j = 0; j < m; j++) {
                intercept -= xMean[j] * coef.get(j);
            }
        }

        if (computeScore && scoresList != null) {
            int actual = 0;
            for (int i = 0; i < maxIter; i++) {
                if (scoresList[i][0] > 0) {
                    actual = i + 1;
                }
            }
            this.scores = new double[actual * 2];
            for (int i = 0; i < actual; i++) {
                this.scores[i * 2] = scoresList[i][0];
                this.scores[i * 2 + 1] = scoresList[i][1];
            }
        }

        this.fitted = true;
        return this;
    }

    private void computePosteriorSigma(Matrix A) {
        try {
            this.sigma = A.inverse();
        } catch (Exception e) {
            this.sigma = null;
        }
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "BayesianRidge");
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
        Validation.checkFitted(fitted, "BayesianRidge");
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
    public double getAlpha() {
        return alpha;
    }
    public double getLambda() {
        return lambda;
    }
    public Matrix getSigma() {
        return sigma;
    }
    public double[] getScores() {
        return scores;
    }
    public int getNIter() {
        return nIter;
    }
    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("max_iter", maxIter); p.put("tol", tol);
        p.put("alpha_1", alpha1); p.put("alpha_2", alpha2);
        p.put("lambda_1", lambda1); p.put("lambda_2", lambda2);
        p.put("fit_intercept", fitIntercept);
        p.put("compute_score", computeScore);
        return Collections.unmodifiableMap(p);
    }
}
