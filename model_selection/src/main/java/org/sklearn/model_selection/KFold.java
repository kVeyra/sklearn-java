package org.sklearn.model_selection;

import java.util.*;

/**
 * K-Folds cross-validator.
 *
 * <p>Provides train/test indices to split data into train/test sets.
 * Each fold is used once as a test set while the remaining k-1 folds
 * form the training set.
 *
 * <p>Mirrors {@code sklearn.model_selection.KFold}.
 */
public class KFold {

    private int nSplits;
    private boolean shuffle;
    private long randomState;

    /**
     * Create KFold.
     *
     * @param nSplits     number of folds (>= 2)
     * @param shuffle     whether to shuffle before splitting
     * @param randomState random seed for shuffling
     */
    public KFold(int nSplits, boolean shuffle, long randomState) {
        if (nSplits < 2) {
            throw new IllegalArgumentException("nSplits must be >= 2");
        }
        this.nSplits = nSplits;
        this.shuffle = shuffle;
        this.randomState = randomState;
    }

    /**
     * Generate indices to split data into train/test sets.
     *
     * @param nSamples number of samples in the data
     * @return list of (trainIndices, testIndices) pairs
     */
    public List<int[][]> split(int nSamples) {
        if (nSamples < nSplits) {
            throw new IllegalArgumentException(
                "nSamples (" + nSamples + ") must be >= nSplits (" + nSplits + ")");
        }

        Integer[] indices = new Integer[nSamples];
        for (int i = 0; i < nSamples; i++) {
            indices[i] = i;
        }

        if (shuffle) {
            Random rng = new Random(randomState);
            Collections.shuffle(Arrays.asList(indices), rng);
        }

        List<int[][]> folds = new ArrayList<>();
        int foldSize = nSamples / nSplits;

        for (int fold = 0; fold < nSplits; fold++) {
            int testStart = fold * foldSize;
            int testEnd = (fold == nSplits - 1) ? nSamples : testStart + foldSize;

            Set<Integer> testSet = new HashSet<>();
            for (int i = testStart; i < testEnd; i++) {
                testSet.add(indices[i]);
            }

            int[] trainIdx = new int[nSamples - (testEnd - testStart)];
            int[] testIdx = new int[testEnd - testStart];
            int ti = 0;
            int tsti = 0;

            for (int i = 0; i < nSamples; i++) {
                if (testSet.contains(indices[i])) {
                    testIdx[tsti++] = indices[i];
                } else {
                    trainIdx[ti++] = indices[i];
                }
            }

            folds.add(new int[][]{trainIdx, testIdx});
        }

        return folds;
    }

    /**
     * Get the number of splits.
     */
    public int getNSplits() {
        return nSplits;
    }
}
