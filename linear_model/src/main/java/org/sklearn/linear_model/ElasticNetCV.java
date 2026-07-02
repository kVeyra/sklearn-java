package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.model_selection.KFold;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Elastic Net with cross-validation.
 */
public class ElasticNetCV implements Predictor<Matrix, Vector, Vector> {

    private double[] l1Ratios;
    private double eps;
    private int nAlphas;
    private boolean fitIntercept;
    private int maxIter;
    private double tol;
    private Integer cv;
    private boolean positive;
    private long randomState;

    private Vector coef;
    private double intercept;
    private double alpha;
    private double l1Ratio;
    private double[] alphas;
    private double[][] msePath;
    private boolean fitted;
    private int nFeatures;

    public ElasticNetCV() {
        this(new double[]{0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0}, 1e-3, 100,
            true, 1000, 1e-4, null, false, 42);
    }

    public ElasticNetCV(double[] l1Ratios, double eps, int nAlphas,
                         boolean fitIntercept, int maxIter, double tol,
                         Integer cv, boolean positive, long randomState) {
        this.l1Ratios = l1Ratios;
        this.eps = eps;
        this.nAlphas = nAlphas;
        this.fitIntercept = fitIntercept;
        this.maxIter = maxIter;
        this.tol = tol;
        this.cv = cv;
        this.positive = positive;
        this.randomState = randomState;
    }

    @Override
    public ElasticNetCV fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        int cvFolds = (cv == null || cv < 2) ? 5 : cv;
        KFold kf = new KFold(cvFolds, true, randomState);

        double bestL1 = l1Ratios[0];
        double bestAlpha = 0;
        double bestMse = Double.POSITIVE_INFINITY;
        this.msePath = new double[l1Ratios.length][];

        for (int lrIdx = 0; lrIdx < l1Ratios.length; lrIdx++) {
            double l1r = l1Ratios[lrIdx];
            double alphaMax = computeAlphaMax(X, y, n, m, l1r);
            double[] alphaGrid = logSpace(alphaMax, alphaMax * eps, nAlphas);
            this.msePath[lrIdx] = new double[alphaGrid.length];

            for (int a = 0; a < alphaGrid.length; a++) {
                double alphaVal = alphaGrid[a];
                double[] foldMse = new double[cvFolds];
                int fold = 0;
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

                    ElasticNet model = new ElasticNet(alphaVal, l1r, fitIntercept, tol, maxIter);
                    model.fit(new Matrix(trainX), new Vector(trainY));

                    Vector pred = model.predict(new Matrix(testX));
                    double mse = 0.0;
                    for (int i = 0; i < testN; i++) {
                        double diff = testY[i] - pred.get(i);
                        mse += diff * diff;
                    }
                    foldMse[fold++] = mse / testN;
                }

                double avgMse = 0.0;
                for (double f : foldMse) {
                    avgMse += f;
                }
                avgMse /= cvFolds;
                this.msePath[lrIdx][a] = avgMse;

                if (avgMse < bestMse) {
                    bestMse = avgMse;
                    bestAlpha = alphaVal;
                    bestL1 = l1r;
                }
            }

            if (alphas == null || alphaGrid.length > (alphas == null ? 0 : alphas.length)) {
                this.alphas = alphaGrid;
            }
        }

        this.alpha = bestAlpha;
        this.l1Ratio = bestL1;

        ElasticNet bestModel = new ElasticNet(bestAlpha, bestL1, fitIntercept, tol, maxIter);
        bestModel.fit(X, y);
        this.coef = bestModel.getCoef();
        this.intercept = bestModel.getIntercept();
        this.fitted = true;
        return this;
    }

    private double computeAlphaMax(Matrix X, Vector y, int n, int m, double l1r) {
        double maxVal = 0.0;
        for (int j = 0; j < m; j++) {
            double dot = 0.0;
            for (int i = 0; i < n; i++) {
                dot += X.get(i, j) * y.get(i);
            }
            maxVal = Math.max(maxVal, Math.abs(dot));
        }
        return maxVal / (n * l1r);
    }

    private double[] logSpace(double start, double end, int n) {
        double[] result = new double[n];
        double logStart = Math.log(start);
        double logEnd = Math.log(end);
        for (int i = 0; i < n; i++) {
            double frac = (double) i / (n - 1);
            result[i] = Math.exp(logStart + frac * (logEnd - logStart));
        }
        return result;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "ElasticNetCV");
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
        Validation.checkFitted(fitted, "ElasticNetCV");
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
    public double getL1Ratio() {
        return l1Ratio;
    }
    public double[] getAlphas() {
        return alphas;
    }
    public double[][] getMsePath() {
        return msePath;
    }
    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("l1_ratio", l1Ratios); p.put("eps", eps);
        p.put("n_alphas", nAlphas); p.put("fit_intercept", fitIntercept);
        p.put("max_iter", maxIter); p.put("tol", tol);
        p.put("cv", cv); p.put("positive", positive);
        return Collections.unmodifiableMap(p);
    }
}
