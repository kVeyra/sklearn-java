package org.sklearn.model_selection;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.*;

/**
 * Split arrays into random train and test subsets.
 *
 * <p>Mirrors {@code sklearn.model_selection.train_test_split}.
 */
public final class TrainTestSplit {

    private TrainTestSplit() {
    }

    /**
     * Result of a train-test split.
     */
    public static class SplitResult {
        public final Matrix xTrain;
        public final Matrix xTest;
        public final Vector yTrain;
        public final Vector yTest;

        public SplitResult(Matrix xTrain, Matrix xTest, Vector yTrain, Vector yTest) {
            this.xTrain = xTrain;
            this.xTest = xTest;
            this.yTrain = yTrain;
            this.yTest = yTest;
        }
    }

    /**
     * Split data into train and test sets.
     *
     * @param X           feature matrix
     * @param y           target vector
     * @param testSize    proportion of test set (e.g., 0.2)
     * @param randomState random seed
     * @return SplitResult containing train/test splits
     */
    public static SplitResult split(Matrix X, Vector y, double testSize, long randomState) {
        return trainTestSplit(X, y, testSize, randomState);
    }

    /**
     * Split data into train and test sets.
     *
     * @param X           feature matrix
     * @param y           target vector
     * @param testSize    proportion of test set (e.g., 0.2)
     * @param randomState random seed
     * @return SplitResult containing train/test splits
     */
    public static SplitResult trainTestSplit(Matrix X, Vector y, double testSize, long randomState) {
        int n = X.rows();
        int nTest = (int) Math.round(n * testSize);
        if (nTest < 1) {
            nTest = 1;
        }
        if (nTest >= n) {
            nTest = n - 1;
        }

        Integer[] boxed = new Integer[n];
        for (int i = 0; i < n; i++) {
            boxed[i] = i;
        }
        Random rng = new Random(randomState);
        Collections.shuffle(Arrays.asList(boxed), rng);

        int nTrain = n - nTest;
        int[] trainIdx = new int[nTrain];
        int[] testIdx = new int[nTest];
        for (int i = 0; i < nTrain; i++) {
            trainIdx[i] = boxed[i];
        }
        for (int i = 0; i < nTest; i++) {
            testIdx[i] = boxed[nTrain + i];
        }

        return new SplitResult(
            extractRows(X, trainIdx),
            extractRows(X, testIdx),
            extractRows(y, trainIdx),
            extractRows(y, testIdx)
        );
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
}
