package org.sklearn.datasets;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.Random;

/**
 * Generate a random n-class classification problem.
 *
 * <p>Mirrors {@code sklearn.datasets.make_classification}.
 */
public final class MakeClassification {

    private MakeClassification() {
    }

    /**
     * Generate a random classification dataset.
     *
     * @param nSamples     number of samples
     * @param nFeatures    total number of features
     * @param nClasses     number of classes
     * @param randomState  random seed
     * @return dataset with data matrix and target vector
     */
    public static Result makeClassification(int nSamples, int nFeatures,
                                            int nClasses, long randomState) {
        return makeClassification(nSamples, nFeatures, nClasses, 0, 1.0, randomState);
    }

    /**
     * Generate classification dataset with full control.
     *
     * @param nSamples        number of samples
     * @param nFeatures       total number of features
     * @param nClasses        number of classes
     * @param nInformative    number of informative features (0 = auto)
     * @param flipY           fraction of labels to randomly flip
     * @param randomState     random seed
     * @return dataset
     */
    public static Result makeClassification(int nSamples, int nFeatures,
                                            int nClasses, int nInformative,
                                            double flipY, long randomState) {
        Random rng = new Random(randomState);
        int nInf = nInformative > 0 ? nInformative : Math.max(1, nFeatures / 2);
        nInf = Math.min(nInf, nFeatures);

        double[][] X = new double[nSamples][nFeatures];
        double[] y = new double[nSamples];

        double[][] centroids = new double[nClasses][nInf];
        for (int c = 0; c < nClasses; c++) {
            for (int j = 0; j < nInf; j++) {
                centroids[c][j] = rng.nextGaussian() * 2;
            }
        }

        int[] classSizes = new int[nClasses];
        int base = nSamples / nClasses;
        int rem = nSamples % nClasses;
        for (int c = 0; c < nClasses; c++) {
            classSizes[c] = base + (c < rem ? 1 : 0);
        }

        int idx = 0;
        for (int c = 0; c < nClasses; c++) {
            for (int i = 0; i < classSizes[c]; i++) {
                for (int j = 0; j < nInf; j++) {
                    X[idx][j] = centroids[c][j] + rng.nextGaussian() * 0.5;
                }
                for (int j = nInf; j < nFeatures; j++) {
                    X[idx][j] = rng.nextGaussian() * 2;
                }
                y[idx] = c;
                idx++;
            }
        }

        if (flipY > 0) {
            for (int i = 0; i < nSamples; i++) {
                if (rng.nextDouble() < flipY) {
                    y[i] = rng.nextInt(nClasses);
                }
            }
        }

        return new Result(new Matrix(X), new Vector(y));
    }

    /**
     * Result of make_classification with data and labels.
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
