package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.model_selection.KFold;
import org.sklearn.utils.Validation;

import java.util.*;

public class RidgeClassifierCV implements Predictor<Matrix, Vector, Vector> {

    private double[] alphas;
    private Integer cv;
    private Ridge bestRidge;
    private double alpha;
    private boolean fitted;
    private int[] classes;
    private int nFeatures;

    public RidgeClassifierCV() {
        this(new double[]{0.1, 1.0, 10.0}, null);
    }

    public RidgeClassifierCV(double[] alphas, Integer cv) {
        this.alphas = alphas;
        this.cv = cv;
    }

    @Override
    public RidgeClassifierCV fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        this.nFeatures = X.cols();

        Set<Integer> unique = new LinkedHashSet<>();
        for (int i = 0; i < y.size(); i++) unique.add((int) y.get(i));
        classes = unique.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(classes);

        if (classes.length != 2) {
            throw new IllegalArgumentException("RidgeClassifierCV only supports binary classification");
        }

        Vector yAdj = new Vector(y.size());
        for (int i = 0; i < y.size(); i++) {
            yAdj.set(i, y.get(i) == classes[0] ? -1.0 : 1.0);
        }

        int n = X.rows();
        int nFolds = (cv == null || cv <= 0) ? n : cv;
        KFold kfold = new KFold(nFolds, true, 42);

        double bestScore = Double.NEGATIVE_INFINITY;
        double bestAlpha = alphas[0];

        for (double a : alphas) {
            List<int[][]> splits = kfold.split(n);
            double totalScore = 0;
            for (int[][] split : splits) {
                int[] trainIdx = split[0];
                int[] testIdx = split[1];
                Matrix Xtr = new Matrix(trainIdx.length, nFeatures);
                Matrix Xte = new Matrix(testIdx.length, nFeatures);
                Vector ytr = new Vector(trainIdx.length);
                Vector yte = new Vector(testIdx.length);
                for (int r = 0; r < trainIdx.length; r++) {
                    for (int c = 0; c < nFeatures; c++) Xtr.set(r, c, X.get(trainIdx[r], c));
                    ytr.set(r, yAdj.get(trainIdx[r]));
                }
                for (int r = 0; r < testIdx.length; r++) {
                    for (int c = 0; c < nFeatures; c++) Xte.set(r, c, X.get(testIdx[r], c));
                    yte.set(r, yAdj.get(testIdx[r]));
                }
                Ridge ridge = new Ridge(a, true);
                ridge.fit(Xtr, ytr);
                Vector preds = ridge.predict(Xte);
                int correct = 0;
                for (int k = 0; k < preds.size(); k++) {
                    if ((preds.get(k) < 0 ? -1.0 : 1.0) == yte.get(k)) correct++;
                }
                totalScore += (double) correct / preds.size();
            }
            double avgScore = totalScore / nFolds;
            if (avgScore > bestScore) {
                bestScore = avgScore;
                bestAlpha = a;
            }
        }

        this.alpha = bestAlpha;
        bestRidge = new Ridge(bestAlpha, true);
        bestRidge.fit(X, yAdj);
        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "RidgeClassifierCV");
        Vector raw = bestRidge.predict(X);
        Vector result = new Vector(X.rows());
        for (int i = 0; i < X.rows(); i++) {
            result.set(i, raw.get(i) < 0 ? classes[0] : classes[1]);
        }
        return result;
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "RidgeClassifierCV");
        Vector preds = predict(X);
        int correct = 0;
        for (int i = 0; i < y.size(); i++) {
            if (preds.get(i) == y.get(i)) correct++;
        }
        return (double) correct / y.size();
    }

    public boolean isFitted() { return fitted; }
    public double getAlpha() { return alpha; }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("alphas", alphas);
        p.put("cv", cv);
        return Collections.unmodifiableMap(p);
    }
}
