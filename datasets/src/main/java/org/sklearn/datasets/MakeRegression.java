package org.sklearn.datasets;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.Random;

/**
 * Generate a random regression problem.
 *
 * <p>Mirrors {@code sklearn.datasets.make_regression}.
 */
public final class MakeRegression {

    private MakeRegression() {
    }

    /**
     * Generate a random regression dataset.
     *
     * @param nSamples    number of samples
     * @param nFeatures   number of features
     * @param noise       standard deviation of Gaussian noise
     * @param randomState random seed
     * @return dataset
     */
    public static Result makeRegression(int nSamples, int nFeatures,
                                        double noise, long randomState) {
        Random rng = new Random(randomState);

        double[][] X = new double[nSamples][nFeatures];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                X[i][j] = rng.nextGaussian();
            }
        }

        double[] trueCoef = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++) {
            trueCoef[j] = rng.nextGaussian();
        }

        double[] y = new double[nSamples];
        for (int i = 0; i < nSamples; i++) {
            double sum = 0;
            for (int j = 0; j < nFeatures; j++) {
                sum += X[i][j] * trueCoef[j];
            }
            y[i] = sum + noise * rng.nextGaussian();
        }

        return new Result(new Matrix(X), new Vector(y));
    }

    /**
     * Result of make_regression with data and targets.
     */
    public static class Result {
        public final Matrix x;
        public final Vector y;
        Result(Matrix x, Vector y) {
            this.x = x;
            this.y = y;
        }
    }
}
