package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Least Angle Regression (LARS) for linear regression.
 *
 * <p>LARS is a regression algorithm for high-dimensional data that
 * builds up the model incrementally by adding features most correlated
 * with the residual.
 *
 * <p>Mirrors {@code sklearn.linear_model.Lars}.
 */
public class Lars implements Predictor<Matrix, Vector, Vector> {

    private int nNonzeroCoefs;
    private boolean fitIntercept;
    private boolean fitted;
    private Vector coef;
    private double intercept;
    private int nFeatures;
    private int[] activeIndices;

    /**
     * Create Lars with default parameters.
     */
    public Lars() {
        this(500, true);
    }

    /**
     * Create Lars with full control.
     *
     * @param nNonzeroCoefs maximum number of non-zero coefficients
     * @param fitIntercept  whether to fit intercept
     */
    public Lars(int nNonzeroCoefs, boolean fitIntercept) {
        this.nNonzeroCoefs = nNonzeroCoefs;
        this.fitIntercept = fitIntercept;
    }

    @Override
    public Lars fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        int n = X.rows();
        int m = X.cols();
        nFeatures = m;

        double[] yc = new double[n];
        double meanY = 0;
        if (fitIntercept) {
            for (int i = 0; i < n; i++) meanY += y.get(i);
            meanY /= n;
        }
        for (int i = 0; i < n; i++) yc[i] = y.get(i) - meanY;
        intercept = meanY;

        double[][] xData = X.toArray();
        double[] residual = yc.clone();
        double[] coefVec = new double[m];
        boolean[] active = new boolean[m];
        int[] activeList = new int[Math.min(nNonzeroCoefs, m)];
        int nActive = 0;

        int maxSteps = Math.min(nNonzeroCoefs, m);
        for (int step = 0; step < maxSteps; step++) {
            double maxCorr = -1;
            int bestIdx = -1;
            for (int j = 0; j < m; j++) {
                if (active[j]) continue;
                double dot = 0;
                for (int i = 0; i < n; i++) {
                    dot += xData[i][j] * residual[i];
                }
                double corr = Math.abs(dot);
                if (corr > maxCorr) {
                    maxCorr = corr;
                    bestIdx = j;
                }
            }
            if (bestIdx < 0 || maxCorr < 1e-15) break;

            active[bestIdx] = true;
            activeList[nActive++] = bestIdx;

            double[][] xActive = new double[n][nActive];
            for (int i = 0; i < n; i++) {
                for (int a = 0; a < nActive; a++) {
                    xActive[i][a] = xData[i][activeList[a]];
                }
            }

            double[][] g = new double[nActive][nActive];
            for (int a = 0; a < nActive; a++) {
                for (int b = 0; b < nActive; b++) {
                    double sum = 0;
                    for (int i = 0; i < n; i++) {
                        sum += xActive[i][a] * xActive[i][b];
                    }
                    g[a][b] = sum;
                }
            }

            double[] s = new double[nActive];
            for (int a = 0; a < nActive; a++) {
                double dot = 0;
                for (int i = 0; i < n; i++) {
                    dot += xData[i][activeList[a]] * residual[i];
                }
                s[a] = dot;
            }

            double[] d = solveLinear(g, s, nActive);

            double[] coefDirection = new double[m];
            for (int a = 0; a < nActive; a++) {
                coefDirection[activeList[a]] = d[a];
            }

            double maxStep = Double.MAX_VALUE;
            for (int j = 0; j < m; j++) {
                if (active[j] || Math.abs(coefDirection[j]) < 1e-15) continue;
                double dotRes = 0;
                double dotDir = 0;
                for (int i = 0; i < n; i++) {
                    dotRes += xData[i][j] * residual[i];
                    dotDir += xData[i][j] * coefDirection[j];
                }
                double stepSize = -dotRes / (dotDir + 1e-15);
                if (stepSize > 0 && stepSize < maxStep) {
                    maxStep = stepSize;
                }
            }

            if (maxStep == Double.MAX_VALUE) {
                maxStep = 1.0;
            }

            for (int j = 0; j < m; j++) {
                coefVec[j] += coefDirection[j] * maxStep;
            }

            for (int i = 0; i < n; i++) {
                double pred = 0;
                for (int j = 0; j < m; j++) {
                    pred += xData[i][j] * coefVec[j];
                }
                residual[i] = yc[i] - pred;
            }
        }

        coef = new Vector(coefVec);
        activeIndices = new int[nActive];
        System.arraycopy(activeList, 0, activeIndices, 0, nActive);
        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "Lars");
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
        return org.sklearn.metrics.RegressionMetrics.r2Score(y, predict(X));
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_nonzero_coefs", nNonzeroCoefs);
        params.put("coef_", coef);
        params.put("intercept_", intercept);
        return params;
    }

    public Vector getCoef() { return coef; }
    public double getIntercept() { return intercept; }
    public int[] getActiveIndices() { return activeIndices; }

    private static double[] solveLinear(double[][] a, double[] b, int n) {
        double[][] aug = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, aug[i], 0, n);
            aug[i][n] = b[i];
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
            if (Math.abs(pivotVal) < 1e-15) continue;
            for (int j = col; j <= n; j++) {
                aug[col][j] /= pivotVal;
            }
            for (int row = 0; row < n; row++) {
                if (row != col) {
                    double factor = aug[row][col];
                    for (int j = col; j <= n; j++) {
                        aug[row][j] -= factor * aug[col][j];
                    }
                }
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = aug[i][n];
        return x;
    }
}
