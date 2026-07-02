package org.sklearn.datasets;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.Random;

/**
 * Make a large circle containing a smaller circle in 2D.
 *
 * <p>Mirrors {@code sklearn.datasets.make_circles}.
 */
public final class MakeCircles {

    private MakeCircles() {
    }

    /**
     * Generate two concentric circles.
     *
     * @param nSamples    total number of samples
     * @param noise       standard deviation of Gaussian noise
     * @param factor      scale factor between inner and outer circle
     * @param randomState random seed
     * @return dataset
     */
    public static Result makeCircles(int nSamples, double noise,
                                     double factor, long randomState) {
        Random rng = new Random(randomState);
        int nSamplesOut = nSamples / 2;
        int nSamplesIn = nSamples - nSamplesOut;

        double[][] X = new double[nSamples][2];
        double[] y = new double[nSamples];

        for (int i = 0; i < nSamplesOut; i++) {
            double t = 2 * Math.PI * rng.nextDouble();
            X[i][0] = Math.cos(t) + rng.nextGaussian() * noise;
            X[i][1] = Math.sin(t) + rng.nextGaussian() * noise;
            y[i] = 0;
        }
        for (int i = 0; i < nSamplesIn; i++) {
            double t = 2 * Math.PI * rng.nextDouble();
            X[nSamplesOut + i][0] = factor * Math.cos(t) + rng.nextGaussian() * noise;
            X[nSamplesOut + i][1] = factor * Math.sin(t) + rng.nextGaussian() * noise;
            y[nSamplesOut + i] = 1;
        }

        return new Result(new Matrix(X), new Vector(y));
    }

    /**
     * Result of make_circles with data and labels.
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
