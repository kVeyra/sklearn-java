package org.sklearn.multiclass;

import org.sklearn.core.Estimator;
import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * One-vs-one multiclass strategy.
 *
 * <p>Fits one binary classifier per pair of classes and predicts
 * by majority vote (most wins).
 *
 * <p>Mirrors {@code sklearn.multiclass.OneVsOneClassifier}.
 */
public class OneVsOneClassifier implements Predictor<Matrix, Vector, Vector> {

    private Estimator<Matrix, Vector> baseEstimator;
    private boolean fitted;
    private int[] classes;
    private int nClasses;
    private int nFeatures;
    private List<Estimator<Matrix, Vector>> estimators;
    private List<int[]> classPairs;

    public OneVsOneClassifier(Estimator<Matrix, Vector> baseEstimator) {
        this.baseEstimator = baseEstimator;
    }

    @Override
    public OneVsOneClassifier fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        nFeatures = X.cols();

        Set<Integer> seen = new LinkedHashSet<>();
        for (int i = 0; i < y.size(); i++) seen.add((int) y.get(i));
        classes = seen.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(classes);
        nClasses = classes.length;

        estimators = new ArrayList<>();
        classPairs = new ArrayList<>();

        for (int a = 0; a < nClasses; a++) {
            for (int b = a + 1; b < nClasses; b++) {
                int cA = classes[a];
                int cB = classes[b];

                List<Integer> pairIndices = new ArrayList<>();
                List<Double> pairTargets = new ArrayList<>();
                for (int i = 0; i < y.size(); i++) {
                    int yi = (int) y.get(i);
                    if (yi == cA || yi == cB) {
                        pairIndices.add(i);
                        pairTargets.add(yi == cA ? 1.0 : 0.0);
                    }
                }

                if (pairIndices.isEmpty()) continue;

                Matrix Xpair = new Matrix(pairIndices.size(), nFeatures);
                Vector ypair = new Vector(pairIndices.size());
                for (int k = 0; k < pairIndices.size(); k++) {
                    int idx = pairIndices.get(k);
                    for (int j = 0; j < nFeatures; j++) {
                        Xpair.set(k, j, X.get(idx, j));
                    }
                    ypair.set(k, pairTargets.get(k));
                }

                Estimator<Matrix, Vector> est = cloneEstimator(baseEstimator);
                est.fit(Xpair, ypair);
                estimators.add(est);
                classPairs.add(new int[]{cA, cB});
            }
        }

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "OneVsOneClassifier");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Expected " + nFeatures + " features, got " + X.cols());
        }

        int n = X.rows();
        int[][] votes = new int[n][nClasses];

        for (int e = 0; e < estimators.size(); e++) {
            @SuppressWarnings("unchecked")
            Predictor<Matrix, Vector, Vector> pred =
                (Predictor<Matrix, Vector, Vector>) estimators.get(e);
            Vector result = pred.predict(X);
            int[] pair = classPairs.get(e);
            int idxA = indexOf(classes, pair[0]);
            int idxB = indexOf(classes, pair[1]);

            for (int i = 0; i < n; i++) {
                if (result.get(i) >= 0.5) {
                    votes[i][idxA]++;
                } else {
                    votes[i][idxB]++;
                }
            }
        }

        double[] preds = new double[n];
        for (int i = 0; i < n; i++) {
            int bestIdx = 0;
            for (int j = 1; j < nClasses; j++) {
                if (votes[i][j] > votes[i][bestIdx]) bestIdx = j;
            }
            preds[i] = classes[bestIdx];
        }
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        return org.sklearn.metrics.ClassificationMetrics.accuracyScore(y, predict(X));
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("classes", classes);
        params.put("n_estimators", estimators.size());
        return params;
    }

    public int[] getClasses() { return classes; }
    public boolean isFitted() { return fitted; }

    public List<Estimator<Matrix, Vector>> getEstimators() { return estimators; }

    private int indexOf(int[] arr, int val) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == val) return i;
        }
        return 0;
    }

    @SuppressWarnings("unchecked")
    private Estimator<Matrix, Vector> cloneEstimator(Estimator<Matrix, Vector> est) {
        try {
            return est.getClass().getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Cannot clone estimator: " + est.getClass().getName(), e);
        }
    }
}
