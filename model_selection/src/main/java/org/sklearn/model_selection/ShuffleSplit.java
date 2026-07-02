package org.sklearn.model_selection;

import java.util.*;

/**
 * Random permutation cross-validator.
 *
 * <p>Yields indices to split data into train/test sets,
 * randomly shuffling and partitioning each time.
 *
 * <p>Mirrors {@code sklearn.model_selection.ShuffleSplit}.
 */
public class ShuffleSplit {

    private int nSplits;
    private double testSize;
    private long randomState;

    /**
     * Create ShuffleSplit.
     *
     * @param nSplits     number of shuffle splits (default 10)
     * @param testSize    proportion of test set (default 0.1)
     * @param randomState random seed
     */
    public ShuffleSplit(int nSplits, double testSize, long randomState) {
        if (nSplits < 1) {
            throw new IllegalArgumentException("nSplits must be >= 1");
        }
        if (testSize <= 0 || testSize >= 1) {
            throw new IllegalArgumentException("testSize must be in (0, 1)");
        }
        this.nSplits = nSplits;
        this.testSize = testSize;
        this.randomState = randomState;
    }

    /**
     * Generate train/test indices.
     *
     * @param nSamples number of samples
     * @return list of (trainIndices, testIndices) pairs
     */
    public List<int[][]> split(int nSamples) {
        if (nSamples < 1) {
            throw new IllegalArgumentException("nSamples must be >= 1");
        }

        List<int[][]> splits = new ArrayList<>();
        Random rng = new Random(randomState);

        for (int s = 0; s < nSplits; s++) {
            Integer[] idx = new Integer[nSamples];
            for (int i = 0; i < nSamples; i++) {
                idx[i] = i;
            }
            Collections.shuffle(Arrays.asList(idx), new Random(rng.nextLong()));

            int nTest = Math.max(1, (int) Math.round(nSamples * testSize));
            if (nTest >= nSamples) {
                nTest = nSamples - 1;
            }
            int nTrain = nSamples - nTest;

            int[] trainIdx = new int[nTrain];
            int[] testIdx = new int[nTest];
            for (int i = 0; i < nTrain; i++) {
                trainIdx[i] = idx[i];
            }
            for (int i = 0; i < nTest; i++) {
                testIdx[i] = idx[nTrain + i];
            }

            splits.add(new int[][]{trainIdx, testIdx});
        }

        return splits;
    }

    public int getNSplits() {
        return nSplits;
    }
}
