package org.sklearn.datasets;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.Random;

/**
 * Make two interleaving half circles.
 *
 * <p>Mirrors {@code sklearn.datasets.make_moons}.
 */
public final class MakeMoons {

    private MakeMoons() {
    }

    /**
     * Generate two interleaving half circles.
     *
     * @param nSamples    total number of samples
     * @param noise       standard deviation of Gaussian noise
     * @param randomState random seed
     * @return dataset
     */
    public static Result makeMoons(int nSamples, double noise, long randomState) {
        Random rng = new Random(randomState);
        int nSamplesOut = nSamples / 2;
        int nSamplesIn = nSamples - nSamplesOut;

        double[][] X = new double[nSamples][2];
        double[] y = new double[nSamples];

        for (int i = 0; i < nSamplesOut; i++) {
            double t = Math.PI * i / (nSamplesOut - 1);
            X[i][0] = Math.cos(t) + rng.nextGaussian() * noise;
            X[i][1] = Math.sin(t) + rng.nextGaussian() * noise;
            y[i] = 0;
        }
        for (int i = 0; i < nSamplesIn; i++) {
            double t = Math.PI * i / (nSamplesIn - 1);
            X[nSamplesOut + i][0] = 1 - Math.cos(t) + rng.nextGaussian() * noise;
            X[nSamplesOut + i][1] = -Math.sin(t) + rng.nextGaussian() * noise + 0.5;
            y[nSamplesOut + i] = 1;
        }

        return new Result(new Matrix(X), new Vector(y));
    }

    /**
     * Result of make_moons with data and labels.
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
