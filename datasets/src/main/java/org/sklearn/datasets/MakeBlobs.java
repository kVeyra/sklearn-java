package org.sklearn.datasets;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.Random;

/**
 * Generate isotropic Gaussian blobs for clustering.
 *
 * <p>Mirrors {@code sklearn.datasets.make_blobs}.
 */
public final class MakeBlobs {

    private MakeBlobs() {
    }

    /**
     * Generate isotropic Gaussian blobs.
     *
     * @param nSamples    number of samples
     * @param nFeatures   number of features
     * @param nCenters    number of centers (classes)
     * @param clusterStd  standard deviation of clusters
     * @param randomState random seed
     * @return dataset with data and labels
     */
    public static Result makeBlobs(int nSamples, int nFeatures, int nCenters,
                                   double clusterStd, long randomState) {
        Random rng = new Random(randomState);

        double[][] centers = new double[nCenters][nFeatures];
        for (int c = 0; c < nCenters; c++) {
            for (int j = 0; j < nFeatures; j++) {
                centers[c][j] = rng.nextGaussian() * 3;
            }
        }

        double[][] X = new double[nSamples][nFeatures];
        double[] y = new double[nSamples];

        int base = nSamples / nCenters;
        int rem = nSamples % nCenters;
        int[] sizes = new int[nCenters];
        for (int c = 0; c < nCenters; c++) {
            sizes[c] = base + (c < rem ? 1 : 0);
        }

        int idx = 0;
        for (int c = 0; c < nCenters; c++) {
            for (int i = 0; i < sizes[c]; i++) {
                for (int j = 0; j < nFeatures; j++) {
                    X[idx][j] = centers[c][j] + rng.nextGaussian() * clusterStd;
                }
                y[idx] = c;
                idx++;
            }
        }

        return new Result(new Matrix(X), new Vector(y));
    }

    /**
     * Result of make_blobs with data and labels.
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
