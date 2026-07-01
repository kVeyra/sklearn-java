package org.sklearn.model_selection;

import org.sklearn.math.Vector;

import java.util.*;

/**
 * Stratified K-Folds cross-validator.
 *
 * <p>Provides train/test indices to split data preserving the percentage
 * of samples for each class.
 *
 * <p>Mirrors {@code sklearn.model_selection.StratifiedKFold}.
 */
public class StratifiedKFold {

    private int nSplits;
    private boolean shuffle;
    private long randomState;

    /**
     * Create StratifiedKFold.
     *
     * @param nSplits     number of folds
     * @param shuffle     whether to shuffle
     * @param randomState random seed
     */
    public StratifiedKFold(int nSplits, boolean shuffle, long randomState) {
        if (nSplits < 2) {
            throw new IllegalArgumentException("nSplits must be >= 2");
        }
        this.nSplits = nSplits;
        this.shuffle = shuffle;
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
        if (n < nSplits) {
            throw new IllegalArgumentException(
                "nSamples (" + n + ") must be >= nSplits (" + nSplits + ")");
        }

        Map<Integer, List<Integer>> classIndices = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) {
            int label = (int) y.get(i);
            classIndices.computeIfAbsent(label, k -> new ArrayList<>()).add(i);
        }

        Random rng = shuffle ? new Random(randomState) : null;
        List<int[][]> folds = new ArrayList<>();

        for (int f = 0; f < nSplits; f++) {
            folds.add(new int[2][]);
        }

        for (Map.Entry<Integer, List<Integer>> entry : classIndices.entrySet()) {
            List<Integer> idxList = new ArrayList<>(entry.getValue());
            if (shuffle) {
                Collections.shuffle(idxList, rng);
            }

            int nClass = idxList.size();
            int baseSize = nClass / nSplits;
            int remainder = nClass % nSplits;

            int pos = 0;
            for (int f = 0; f < nSplits; f++) {
                int foldSize = baseSize + (f < remainder ? 1 : 0);
                int[] foldTrain = folds.get(f)[0];
                int[] foldTest = folds.get(f)[1];

                int[] testPart = new int[foldSize];
                for (int i = 0; i < foldSize; i++) {
                    testPart[i] = idxList.get(pos++);
                }

                int[] trainPart = new int[nClass - foldSize];
                int ti = 0;
                for (int i = 0; i < nClass; i++) {
                    boolean isTest = false;
                    for (int t : testPart) {
                        if (idxList.get(i) == t) {
                            isTest = true;
                            break;
                        }
                    }
                    if (!isTest) {
                        trainPart[ti++] = idxList.get(i);
                    }
                }

                int[] prevTrain = folds.get(f)[0];
                int[] prevTest = folds.get(f)[1];
                int[] mergedTrain = new int[(prevTrain != null ? prevTrain.length : 0) + trainPart.length];
                int[] mergedTest = new int[(prevTest != null ? prevTest.length : 0) + testPart.length];
                int dst = 0;
                if (prevTrain != null) {
                    System.arraycopy(prevTrain, 0, mergedTrain, 0, prevTrain.length);
                    dst = prevTrain.length;
                }
                System.arraycopy(trainPart, 0, mergedTrain, dst, trainPart.length);
                dst = 0;
                if (prevTest != null) {
                    System.arraycopy(prevTest, 0, mergedTest, 0, prevTest.length);
                    dst = prevTest.length;
                }
                System.arraycopy(testPart, 0, mergedTest, dst, testPart.length);

                folds.set(f, new int[][]{mergedTrain, mergedTest});
            }
        }

        return folds;
    }
}
