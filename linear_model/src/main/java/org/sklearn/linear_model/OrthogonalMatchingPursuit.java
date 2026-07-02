package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.metrics.RegressionMetrics;
import org.sklearn.utils.Validation;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Orthogonal Matching Pursuit (OMP) for linear regression.
 *
 * <p>Selects nNonzeroCoefs features by greedily minimizing the
 * residual. Equivalent to a greedy forward feature selection
 * with orthogonal projection.
 *
 * <p>Mirrors {@code sklearn.linear_model.OrthogonalMatchingPursuit}.
 *
 * <p>Usage:
 * <pre>{@code
 * OrthogonalMatchingPursuit omp = new OrthogonalMatchingPursuit(5);
 * omp.fit(X, y);
 * Vector preds = omp.predict(X_test);
 * }</pre>
 */
public class OrthogonalMatchingPursuit implements Predictor<Matrix, Vector, Vector> {

    private int nNonzeroCoefs;
    private boolean fitted;
    private Vector coef;
    private double intercept;
    private int nFeatures;
    private int[] activeIndices;

    /**
     * Create OrthogonalMatchingPursuit.
     *
     * @param nNonzeroCoefs desired number of non-zero coefficients
     */
    public OrthogonalMatchingPursuit(int nNonzeroCoefs) {
        if (nNonzeroCoefs < 1) {
            throw new IllegalArgumentException("nNonzeroCoefs must be >= 1");
        }
        this.nNonzeroCoefs = nNonzeroCoefs;
    }

    @Override
    public OrthogonalMatchingPursuit fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        int n = X.rows();
        int m = X.cols();
        nFeatures = m;

        double meanY = 0;
        for (int i = 0; i < n; i++) meanY += y.get(i);
        meanY /= n;

        intercept = meanY;

        double[] yc = new double[n];
        for (int i = 0; i < n; i++) {
            yc[i] = y.get(i) - meanY;
        }

        double[][] xData = X.toArray();
        double[] residual = yc.clone();
        boolean[] selected = new boolean[m];
        int[] active = new int[Math.min(nNonzeroCoefs, m)];
        int nActive = 0;
        double[] finalBeta = new double[0];

        for (int iter = 0; iter < Math.min(nNonzeroCoefs, m); iter++) {
            double maxCorr = -1;
            int bestIdx = -1;

            for (int j = 0; j < m; j++) {
                if (selected[j]) continue;
                double dot = 0;
                for (int i = 0; i < n; i++) {
                    dot += xData[i][j] * residual[i];
                }
                double norm = 0;
                for (int i = 0; i < n; i++) {
                    norm += xData[i][j] * xData[i][j];
                }
                double corr = Math.abs(dot) / Math.sqrt(norm);
                if (corr > maxCorr) {
                    maxCorr = corr;
                    bestIdx = j;
                }
            }

            if (bestIdx < 0) break;
            selected[bestIdx] = true;
            active[nActive++] = bestIdx;

            double[][] xSub = new double[n][nActive];
            for (int i = 0; i < n; i++) {
                for (int a = 0; a < nActive; a++) {
                    xSub[i][a] = xData[i][active[a]];
                }
            }

            double[][] xtx = new double[nActive][nActive];
            double[] xty = new double[nActive];
            for (int a = 0; a < nActive; a++) {
                for (int b = 0; b < nActive; b++) {
                    double sum = 0;
                    for (int i = 0; i < n; i++) {
                        sum += xSub[i][a] * xSub[i][b];
                    }
                    xtx[a][b] = sum;
                }
                double sum = 0;
                for (int i = 0; i < n; i++) {
                    sum += xSub[i][a] * yc[i];
                }
                xty[a] = sum;
            }

            finalBeta = solveLinear(xtx, xty, nActive);

            for (int i = 0; i < n; i++) {
                double pred = 0;
                for (int a = 0; a < nActive; a++) {
                    pred += xSub[i][a] * finalBeta[a];
                }
                residual[i] = yc[i] - pred;
            }
        }

        double[] coefVec = new double[m];
        for (int a = 0; a < nActive; a++) {
            coefVec[active[a]] = finalBeta[a];
        }

        coef = new Vector(coefVec);
        activeIndices = new int[nActive];
        System.arraycopy(active, 0, activeIndices, 0, nActive);
        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "OrthogonalMatchingPursuit");
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
        params.put("n_nonzero_coefs", nNonzeroCoefs);
        params.put("coef_", coef);
        params.put("intercept_", intercept);
        params.put("active_", activeIndices);
        return params;
    }

    public Vector getCoef() {
        return coef;
    }

    public double getIntercept() {
        return intercept;
    }

    public int[] getActiveIndices() {
        return activeIndices;
    }

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
        for (int i = 0; i < n; i++) {
            x[i] = aug[i][n];
        }
        return x;
    }
}
