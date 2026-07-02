package org.sklearn.model_selection;

import org.sklearn.math.Vector;

import java.util.*;

/**
 * Stratified ShuffleSplit cross-validator.
 *
 * <p>Provides train/test indices to split data, preserving the percentage
 * of samples for each class in each split.
 *
 * <p>Mirrors {@code sklearn.model_selection.StratifiedShuffleSplit}.
 */
public class StratifiedShuffleSplit {

    private int nSplits;
    private double testSize;
    private long randomState;

    /**
     * Create StratifiedShuffleSplit.
     *
     * @param nSplits     number of splits (default 10)
     * @param testSize    proportion of test set (default 0.1)
     * @param randomState random seed
     */
    public StratifiedShuffleSplit(int nSplits, double testSize, long randomState) {
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
     * Generate stratified train/test indices.
     *
     * @param y target labels
     * @return list of (trainIndices, testIndices) pairs
     */
    public List<int[][]> split(Vector y) {
        int n = y.size();
        if (n < 1) {
            throw new IllegalArgumentException("nSamples must be >= 1");
        }

        Map<Integer, List<Integer>> classIndices = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) {
            int label = (int) y.get(i);
            classIndices.computeIfAbsent(label, k -> new ArrayList<>()).add(i);
        }

        List<int[][]> splits = new ArrayList<>();
        Random masterRng = new Random(randomState);

        for (int s = 0; s < nSplits; s++) {
            Random splitRng = new Random(masterRng.nextLong());
            int[] trainIdx = new int[0];
            int[] testIdx = new int[0];

            for (Map.Entry<Integer, List<Integer>> entry : classIndices.entrySet()) {
                List<Integer> idxList = new ArrayList<>(entry.getValue());
                Collections.shuffle(idxList, new Random(splitRng.nextLong()));

                int nClass = idxList.size();
                int nTest = Math.max(1, (int) Math.round(nClass * testSize));
                if (nTest >= nClass) {
                    nTest = nClass - 1;
                }
                int nTrain = nClass - nTest;

                int[] classTrain = new int[nTrain];
                int[] classTest = new int[nTest];
                for (int i = 0; i < nTrain; i++) {
                    classTrain[i] = idxList.get(i);
                }
                for (int i = 0; i < nTest; i++) {
                    classTest[i] = idxList.get(nTrain + i);
                }

                trainIdx = merge(trainIdx, classTrain);
                testIdx = merge(testIdx, classTest);
            }

            splits.add(new int[][]{trainIdx, testIdx});
        }

        return splits;
    }

    private static int[] merge(int[] a, int[] b) {
        int[] result = new int[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }

    public int getNSplits() {
        return nSplits;
    }
}
