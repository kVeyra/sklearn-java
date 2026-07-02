package org.sklearn.model_selection;

import org.sklearn.core.Estimator;
import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Cross-validation utilities.
 *
 * <p>Mirrors {@code sklearn.model_selection.cross_val_score}.
 */
public final class CrossValidation {

    private CrossValidation() {
    }

    /**
     * Evaluate a score by cross-validation.
     *
     * @param estimator the estimator to fit
     * @param X         training data
     * @param y         target values
     * @param cv        the cross-validation splitter
     * @return array of scores for each fold
     */
    @SuppressWarnings("unchecked")
    public static double[] crossValScore(Estimator<Matrix, Vector> estimator,
                                          Matrix X, Vector y, KFold cv) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        List<int[][]> folds = cv.split(X.rows());
        double[] scores = new double[folds.size()];

        for (int f = 0; f < folds.size(); f++) {
            int[][] fold = folds.get(f);
            int[] trainIdx = fold[0];
            int[] testIdx = fold[1];

            Matrix xTrain = extractRows(X, trainIdx);
            Vector yTrain = extractRows(y, trainIdx);
            Matrix xTest = extractRows(X, testIdx);
            Vector yTest = extractRows(y, testIdx);

            Estimator<Matrix, Vector> clone = cloneEstimator(estimator);
            clone.fit(xTrain, yTrain);

            if (clone instanceof Predictor) {
                scores[f] = ((Predictor<Matrix, Vector, ?>) clone).score(xTest, yTest);
            }
        }

        return scores;
    }

    /**
     * Cross-validate with a StratifiedKFold splitter.
     */
    public static double[] crossValScore(Estimator<Matrix, Vector> estimator,
                                          Matrix X, Vector y, StratifiedKFold cv) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        List<int[][]> folds = cv.split(y);
        double[] scores = new double[folds.size()];

        for (int f = 0; f < folds.size(); f++) {
            int[][] fold = folds.get(f);
            int[] trainIdx = fold[0];
            int[] testIdx = fold[1];

            Matrix xTrain = extractRows(X, trainIdx);
            Vector yTrain = extractRows(y, trainIdx);
            Matrix xTest = extractRows(X, testIdx);
            Vector yTest = extractRows(y, testIdx);

            Estimator<Matrix, Vector> clone = cloneEstimator(estimator);
            clone.fit(xTrain, yTrain);

            if (clone instanceof Predictor) {
                scores[f] = ((Predictor<Matrix, Vector, ?>) clone).score(xTest, yTest);
            }
        }

        return scores;
    }

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

    @SuppressWarnings("unchecked")
    private static <D, T> Estimator<D, T> cloneEstimator(Estimator<D, T> est) {
        try {
            return est.getClass().getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Cannot clone estimator: " + est.getClass().getName(), e);
        }
    }
}
