package org.sklearn.model_selection;

import java.util.*;

public class RepeatedKFold {

    private int nSplits;
    private int nRepeats;
    private long randomState;

    public RepeatedKFold(int nSplits, int nRepeats, long randomState) {
        this.nSplits = nSplits;
        this.nRepeats = nRepeats;
        this.randomState = randomState;
    }

    public List<int[][]> split(int nSamples) {
        List<int[][]> allFolds = new ArrayList<>();
        for (int rep = 0; rep < nRepeats; rep++) {
            KFold kf = new KFold(nSplits, true, randomState + rep * 1000L);
            allFolds.addAll(kf.split(nSamples));
        }
        return allFolds;
    }

    public int getNSplits() { return nSplits * nRepeats; }
}
