package org.sklearn.model_selection;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.math.RandomGenerator;

/**
 * Train-test splitter.
 *
 * <p>Splits data into random train and test subsets.
 * Mirrors {@code sklearn.model_selection.train_test_split}.
 */
public final class TrainTestSplit {

    private TrainTestSplit() {
    }

    /**
     * Result of a train-test split.
     */
    public static class Split {
        public final Matrix xTrain;
        public final Matrix xTest;
        public final Vector yTrain;
        public final Vector yTest;

        Split(Matrix xTrain, Matrix xTest, Vector yTrain, Vector yTest) {
            this.xTrain = xTrain;
            this.xTest = xTest;
            this.yTrain = yTrain;
            this.yTest = yTest;
        }
    }

    /**
     * Split arrays into random train and test subsets.
     *
     * @param X        samples, shape (n_samples, n_features)
     * @param y        targets, shape (n_samples,)
     * @param testSize fraction of data to include in test split
     * @param seed     random seed
     * @return Split with train/test subsets
     */
    public static Split split(Matrix X, Vector y, double testSize, long seed) {
        int n = X.rows();
        int m = X.cols();
        int nTest = Math.max(1, (int) Math.round(n * testSize));
        int nTrain = n - nTest;

        RandomGenerator rng = new RandomGenerator(seed);
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        // Fisher-Yates shuffle
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }

        Matrix xTrain = new Matrix(nTrain, m);
        Matrix xTest = new Matrix(nTest, m);
        Vector yTrain = new Vector(nTrain);
        Vector yTest = new Vector(nTest);

        for (int i = 0; i < nTrain; i++) {
            int idx = indices[i];
            for (int j = 0; j < m; j++) {
                xTrain.set(i, j, X.get(idx, j));
            }
            yTrain.set(i, y.get(idx));
        }
        for (int i = 0; i < nTest; i++) {
            int idx = indices[nTrain + i];
            for (int j = 0; j < m; j++) {
                xTest.set(i, j, X.get(idx, j));
            }
            yTest.set(i, y.get(idx));
        }

        return new Split(xTrain, xTest, yTrain, yTest);
    }
}
