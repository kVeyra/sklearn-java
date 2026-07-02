package org.sklearn.model_selection;

import java.util.*;

public class LeaveOneOut {

    public List<int[][]> split(int nSamples) {
        List<int[][]> folds = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            int[] trainIdx = new int[nSamples - 1];
            int[] testIdx = new int[]{i};
            int pos = 0;
            for (int j = 0; j < nSamples; j++) {
                if (j != i) trainIdx[pos++] = j;
            }
            folds.add(new int[][]{trainIdx, testIdx});
        }
        return folds;
    }

    public int getNSplits() { return -1; }
}
