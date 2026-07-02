package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Logistic Regression with built-in cross-validation.
 *
 * <p>Automatically selects the best regularization parameter C
 * via cross-validation over a grid of values.
 *
 * <p>Mirrors {@code sklearn.linear_model.LogisticRegressionCV}.
 */
public class LogisticRegressionCV implements Predictor<Matrix, Vector, Vector> {

    private int[] cs;
    private int cv;
    private double tol;
    private int maxIter;
    private boolean fitIntercept;
    private boolean fitted;
    private double bestC;
    private Vector coef;
    private Vector intercept;
    private int[] classes;
    private int nFeatures;
    private LogisticRegression bestEstimator;

    /**
     * Create LogisticRegressionCV with default params.
     */
    public LogisticRegressionCV() {
        this(new int[]{10}, 3, true, 1e-4, 100);
    }

    /**
     * Create LogisticRegressionCV.
     *
     * @param cs            C values to try (each 1/C is regularization)
     * @param cv            number of CV folds
     * @param fitIntercept  whether to fit intercept
     * @param tol           convergence tolerance
     * @param maxIter       max iterations
     */
    public LogisticRegressionCV(int[] cs, int cv, boolean fitIntercept,
                                double tol, int maxIter) {
        this.cs = cs;
        this.cv = cv;
        this.fitIntercept = fitIntercept;
        this.tol = tol;
        this.maxIter = maxIter;
    }

    @Override
    public LogisticRegressionCV fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        nFeatures = X.cols();

        Set<Integer> seen = new LinkedHashSet<>();
        for (int i = 0; i < y.size(); i++) seen.add((int) y.get(i));
        classes = seen.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(classes);

        org.sklearn.model_selection.KFold kf =
            new org.sklearn.model_selection.KFold(cv, false, 42);

        double bestScore = -Double.MAX_VALUE;

        for (double C : cs) {
            double[] foldScores = new double[cv];
            List<int[][]> folds = kf.split(X.rows());

            for (int f = 0; f < folds.size(); f++) {
                int[][] fold = folds.get(f);
                int[] trainIdx = fold[0];
                int[] testIdx = fold[1];

                Matrix xTrain = extractRows(X, trainIdx);
                Vector yTrain = extractRows(y, trainIdx);
                Matrix xTest = extractRows(X, testIdx);
                Vector yTest = extractRows(y, testIdx);

                LogisticRegression lr = new LogisticRegression(C, fitIntercept, tol, maxIter);
                lr.fit(xTrain, yTrain);
                foldScores[f] = lr.score(xTest, yTest);
            }

            double avgScore = 0;
            for (double s : foldScores) avgScore += s;
            avgScore /= foldScores.length;

            if (avgScore > bestScore) {
                bestScore = avgScore;
                bestC = C;
            }
        }

        bestEstimator = new LogisticRegression(bestC, fitIntercept, tol, maxIter);
        bestEstimator.fit(X, y);
        coef = bestEstimator.getCoef();
        intercept = bestEstimator.getIntercept();

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "LogisticRegressionCV");
        return bestEstimator.predict(X);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "LogisticRegressionCV");
        return bestEstimator.score(X, y);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("C_", bestC);
        params.put("coef_", coef);
        params.put("intercept_", intercept);
        params.put("classes_", classes);
        return params;
    }

    public Vector getCoef() { return coef; }
    public Vector getIntercept() { return intercept; }
    public int[] getClasses() { return classes; }
    public double getBestC() { return bestC; }

    private static Matrix extractRows(Matrix X, int[] indices) {
        int n = indices.length;
        int m = X.cols();
        double[][] data = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                data[i][j] = X.get(indices[i], j);
            }
        }
        return new Matrix(data);
    }

    private static Vector extractRows(Vector y, int[] indices) {
        int n = indices.length;
        double[] data = new double[n];
        for (int i = 0; i < n; i++) {
            data[i] = y.get(indices[i]);
        }
        return new Vector(data);
    }
}
