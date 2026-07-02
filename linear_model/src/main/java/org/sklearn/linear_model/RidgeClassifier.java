package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

public class RidgeClassifier implements Predictor<Matrix, Vector, Vector> {

    private Ridge ridge;
    private double alpha;
    private boolean fitIntercept;
    private boolean fitted;
    private int[] classes;
    private int nFeatures;

    public RidgeClassifier() {
        this(1.0, true);
    }

    public RidgeClassifier(double alpha, boolean fitIntercept) {
        this.alpha = alpha;
        this.fitIntercept = fitIntercept;
    }

    @Override
    public RidgeClassifier fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        this.nFeatures = X.cols();

        Set<Integer> unique = new LinkedHashSet<>();
        for (int i = 0; i < y.size(); i++) unique.add((int) y.get(i));
        classes = unique.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(classes);

        if (classes.length == 2) {
            Vector yAdj = new Vector(y.size());
            for (int i = 0; i < y.size(); i++) {
                yAdj.set(i, y.get(i) == classes[0] ? -1.0 : 1.0);
            }
            ridge = new Ridge(alpha, fitIntercept);
            ridge.fit(X, yAdj);
        } else {
            throw new IllegalArgumentException("RidgeClassifier only supports binary classification");
        }

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "RidgeClassifier");
        Vector raw = ridge.predict(X);
        Vector result = new Vector(X.rows());
        for (int i = 0; i < X.rows(); i++) {
            result.set(i, raw.get(i) < 0 ? classes[0] : classes[1]);
        }
        return result;
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "RidgeClassifier");
        Vector preds = predict(X);
        int correct = 0;
        for (int i = 0; i < y.size(); i++) {
            if (preds.get(i) == y.get(i)) correct++;
        }
        return (double) correct / y.size();
    }

    public boolean isFitted() { return fitted; }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("alpha", alpha);
        p.put("fit_intercept", fitIntercept);
        return Collections.unmodifiableMap(p);
    }
}
