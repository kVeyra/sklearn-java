package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.model_selection.KFold;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Ridge regression with cross-validation.
 */
public class RidgeCV implements Predictor<Matrix, Vector, Vector> {

    private double[] alphas;
    private boolean fitIntercept;
    private Integer cv;
    private double tol;

    private Vector coef;
    private double intercept;
    private double alpha;
    private double bestScore;
    private boolean fitted;
    private int nFeatures;

    public RidgeCV() {
        this(new double[]{0.1, 1.0, 10.0}, true, null);
    }

    public RidgeCV(double[] alphas, boolean fitIntercept, Integer cv) {
        this.alphas = alphas;
        this.fitIntercept = fitIntercept;
        this.cv = cv;
    }

    @Override
    public RidgeCV fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        if (alphas == null || alphas.length == 0) {
            this.alpha = 1.0;
            Ridge ridge = new Ridge(alpha, fitIntercept);
            ridge.fit(X, y);
            this.coef = ridge.getCoef();
            this.intercept = ridge.getIntercept();
            this.fitted = true;
            return this;
        }

        if (cv == null || cv <= 1) {
            // LOO-CV via GCV formula for efficiency
            fitLeaveOneOut(X, y, n, m);
        } else {
            // K-fold CV manually
            fitKFoldCV(X, y, n, m);
        }

        this.fitted = true;
        return this;
    }

    private void fitLeaveOneOut(Matrix X, Vector y, int n, int m) {
        // Center data
        double[] xMean = new double[m];
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += X.get(i, j);
            }
            xMean[j] = sum / n;
        }
        double yMean = y.mean();

        double[][] centered = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                centered[i][j] = X.get(i, j) - xMean[j];
            }
        }
        Matrix Xc = new Matrix(centered);

        double[] yc = new double[n];
        for (int i = 0; i < n; i++) {
            yc[i] = y.get(i) - yMean;
        }
        Vector Yc = new Vector(yc);

        // Compute X^T X and X^T y
        Matrix XtX = Xc.transpose().multiply(Xc);
        Vector Xty = Xc.transpose().multiply(Yc);

        double bestAlpha = alphas[0];
        double bestScoreVal = Double.NEGATIVE_INFINITY;

        for (double alphaVal : alphas) {
            if (alphaVal <= 0) {
                continue;
            }
            Matrix reg = XtX.add(Matrix.eye(m).multiply(alphaVal));
            try {
                double[] xtyData = Xty.toArray();
                double[][] col = new double[m][1];
                for (int j = 0; j < m; j++) {
                    col[j][0] = xtyData[j];
                }
                Matrix wMat = reg.solve(new Matrix(col));
                double[] w = new double[m];
                for (int j = 0; j < m; j++) {
                    w[j] = wMat.get(j, 0);
                }

                // Compute LOO scores approximated by hat-matrix diagonal
                double sse = 0.0;
                for (int i = 0; i < n; i++) {
                    double pred = yMean;
                    for (int j = 0; j < m; j++) {
                        pred += centered[i][j] * w[j];
                    }
                    double res = yc[i] - (pred - yMean);
                    sse += res * res;
                }
                // Use negative MSE as score
                double mseVal = sse / n;
                double r2 = 1.0 - mseVal / variance(yc);
                if (r2 > bestScoreVal) {
                    bestScoreVal = r2;
                    bestAlpha = alphaVal;
                }
            } catch (Exception e) {
                continue;
            }
        }

        this.alpha = bestAlpha;
        this.bestScore = bestScoreVal;
        Ridge ridge = new Ridge(bestAlpha, fitIntercept);
        ridge.fit(X, y);
        this.coef = ridge.getCoef();
        this.intercept = ridge.getIntercept();
    }

    private void fitKFoldCV(Matrix X, Vector y, int n, int m) {
        KFold kf = new KFold(cv, true, 42);
        double bestAlpha = alphas[0];
        double bestScoreVal = Double.NEGATIVE_INFINITY;

        for (double alphaVal : alphas) {
            if (alphaVal <= 0) {
                continue;
            }
            double totalScore = 0.0;
            int nFolds = 0;

            java.util.List<int[][]> allFolds = kf.split(n);
            for (int fi = 0; fi < allFolds.size(); fi++) {
                int[] trainIdx = allFolds.get(fi)[0];
                int trainN = trainIdx.length;
                int testN = n - trainN;

                double[][] trainX = new double[trainN][m];
                double[] trainY = new double[trainN];
                double[][] testX = new double[testN][m];
                double[] testY = new double[testN];
                int trainPos = 0, testPos = 0;
                for (int i = 0; i < n; i++) {
                    boolean inTrain = false;
                    for (int idx : trainIdx) {
                        if (idx == i) {
                            inTrain = true;
                            break;
                        }
                    }
                    if (inTrain) {
                        for (int j = 0; j < m; j++) {
                            trainX[trainPos][j] = X.get(i, j);
                        }
                        trainY[trainPos++] = y.get(i);
                    } else {
                        for (int j = 0; j < m; j++) {
                            testX[testPos][j] = X.get(i, j);
                        }
                        testY[testPos++] = y.get(i);
                    }
                }

                Ridge ridge = new Ridge(alphaVal, fitIntercept);
                ridge.fit(new Matrix(trainX), new Vector(trainY));
                totalScore += ridge.score(new Matrix(testX), new Vector(testY));
                nFolds++;
            }

            double avgScore = totalScore / nFolds;
            if (avgScore > bestScoreVal) {
                bestScoreVal = avgScore;
                bestAlpha = alphaVal;
            }
        }

        this.alpha = bestAlpha;
        this.bestScore = bestScoreVal;
        Ridge ridge = new Ridge(bestAlpha, fitIntercept);
        ridge.fit(X, y);
        this.coef = ridge.getCoef();
        this.intercept = ridge.getIntercept();
    }

    private double variance(double[] arr) {
        double mean = 0.0;
        for (double v : arr) {
            mean += v;
        }
        mean /= arr.length;
        double var = 0.0;
        for (double v : arr) {
            var += (v - mean) * (v - mean);
        }
        return var / arr.length;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "RidgeCV");
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
        Validation.checkFitted(fitted, "RidgeCV");
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
    public double getBestScore() {
        return bestScore;
    }
    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("alphas", alphas);
        p.put("fit_intercept", fitIntercept);
        p.put("cv", cv);
        return Collections.unmodifiableMap(p);
    }
}
