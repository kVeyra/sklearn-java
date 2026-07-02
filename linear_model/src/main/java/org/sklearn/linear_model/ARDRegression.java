package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.metrics.RegressionMetrics;
import org.sklearn.utils.Validation;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Bayesian ARD (Automatic Relevance Determination) regression.
 *
 * <p>Fits a Bayesian linear regression model with ARD prior,
 * allowing each weight to have its own precision parameter
 * (automatic relevance determination).
 *
 * <p>Mirrors {@code sklearn.linear_model.ARDRegression}.
 *
 * <p>Usage:
 * <pre>{@code
 * ARDRegression ard = new ARDRegression();
 * ard.fit(X, y);
 * Vector preds = ard.predict(X_test);
 * }</pre>
 */
public class ARDRegression implements Predictor<Matrix, Vector, Vector> {

    private int maxIter;
    private double tol;
    private double alpha1;
    private double alpha2;
    private double lambda1;
    private double lambda2;
    private double thresholdLambda;
    private boolean fitted;
    private Vector coef;
    private double intercept;
    private Vector sigma;
    private double alpha;
    private double lambda;
    private int nFeatures;

    /**
     * Create ARDRegression with default parameters.
     */
    public ARDRegression() {
        this(300, 1e-3, 1e-6, 1e-6, 1e-6, 1e-6, 1e4);
    }

    /**
     * Create ARDRegression with full parameter control.
     */
    public ARDRegression(int maxIter, double tol,
                         double alpha1, double alpha2,
                         double lambda1, double lambda2,
                         double thresholdLambda) {
        this.maxIter = maxIter;
        this.tol = tol;
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
        this.thresholdLambda = thresholdLambda;
    }

    @Override
    public ARDRegression fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        int n = X.rows();
        int m = X.cols();
        nFeatures = m;

        double[][] xData = X.toArray();
        double[] yData = y.toArray();

        double meanY = 0;
        for (double v : yData) meanY += v;
        meanY /= n;
        intercept = meanY;

        double[] yc = new double[n];
        for (int i = 0; i < n; i++) {
            yc[i] = yData[i] - meanY;
        }

        double[] coefVec = new double[m];
        double[] alphaVec = new double[m];
        double lambdaVal = lambda1 / lambda2;
        double alphaVal = alpha1 / alpha2;

        for (int i = 0; i < m; i++) {
            alphaVec[i] = 1.0;
        }

        double[][] xtx = new double[m][m];
        double[] xty = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += xData[k][i] * xData[k][j];
                }
                xtx[i][j] = sum;
            }
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += xData[k][i] * yc[k];
            }
            xty[i] = sum;
        }

        for (int iter = 0; iter < maxIter; iter++) {
            double[][] gram = new double[m][m];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    gram[i][j] = lambdaVal * xtx[i][j];
                }
                gram[i][i] += alphaVec[i];
            }

            double[][] invGram = invert(gram, m);
            double[] newCoef = new double[m];
            for (int i = 0; i < m; i++) {
                double sum = 0;
                for (int j = 0; j < m; j++) {
                    sum += invGram[i][j] * xty[j];
                }
                newCoef[i] = lambdaVal * sum;
            }

            double[] gamma = new double[m];
            for (int i = 0; i < m; i++) {
                gamma[i] = 1 - alphaVec[i] * invGram[i][i];
            }

            double newLambda = 0;
            double sumGamma = 0;
            for (int i = 0; i < m; i++) {
                double ri = yc[i];
                for (int j = 0; j < m; j++) {
                    ri -= xData[i][j] * newCoef[j];
                }
                newLambda += ri * ri;
                sumGamma += gamma[i];
            }
            newLambda = (n - sumGamma + 2 * lambda1) / (newLambda + 2 * lambda2);

            double maxAlphaChange = 0;
            for (int i = 0; i < m; i++) {
                double newAlpha = (gamma[i] + 2 * alpha1) / (newCoef[i] * newCoef[i] + 2 * alpha2);
                double change = Math.abs(alphaVec[i] - newAlpha);
                if (change > maxAlphaChange) maxAlphaChange = change;
                alphaVec[i] = newAlpha;
            }

            System.arraycopy(newCoef, 0, coefVec, 0, m);
            lambdaVal = newLambda;
            alphaVal = 0;
            int nActive = 0;
            for (int i = 0; i < m; i++) {
                alphaVal += alphaVec[i];
                if (alphaVec[i] < thresholdLambda) nActive++;
            }
            alphaVal /= m;

            if (maxAlphaChange < tol) break;
        }

        coef = new Vector(coefVec);
        alpha = alphaVal;
        lambda = lambdaVal;

        sigma = new Vector(m);
        double[][] gram = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                gram[i][j] = lambda * xtx[i][j];
            }
            gram[i][i] += alphaVec[i];
        }
        double[][] invGram = invert(gram, m);
        for (int i = 0; i < m; i++) {
            sigma.set(i, Math.sqrt(invGram[i][i]));
        }

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "ARDRegression");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Expected " + nFeatures + " features, got " + X.cols());
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
        return RegressionMetrics.r2Score(y, predict(X));
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("coef_", coef);
        params.put("intercept_", intercept);
        params.put("alpha_", alpha);
        params.put("lambda_", lambda);
        params.put("sigma_", sigma);
        return params;
    }

    public Vector getCoef() {
        return coef;
    }

    public double getIntercept() {
        return intercept;
    }

    // ---- helpers ----

    private static double[][] invert(double[][] a, int n) {
        double[][] aug = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aug[i][j] = a[i][j];
            }
            aug[i][n + i] = 1;
        }

        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) {
                    pivot = row;
                }
            }
            double[] tmp = aug[col];
            aug[col] = aug[pivot];
            aug[pivot] = tmp;

            double pivotVal = aug[col][col];
            for (int j = 0; j < 2 * n; j++) {
                aug[col][j] /= pivotVal;
            }
            for (int row = 0; row < n; row++) {
                if (row != col) {
                    double factor = aug[row][col];
                    for (int j = 0; j < 2 * n; j++) {
                        aug[row][j] -= factor * aug[col][j];
                    }
                }
            }
        }

        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inv[i][j] = aug[i][n + j];
            }
        }
        return inv;
    }
}
