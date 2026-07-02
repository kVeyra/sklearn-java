package org.sklearn.dummy;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

public class DummyClassifier implements Predictor<Matrix, Vector, Vector> {

    private String strategy;
    private Double constant;
    private boolean fitted;
    private int[] classes;
    private double[] classPrior;
    private int nFeatures;

    public DummyClassifier() {
        this("prior", null);
    }

    public DummyClassifier(String strategy, Double constant) {
        if (!strategy.equals("prior") && !strategy.equals("most_frequent")
            && !strategy.equals("stratified") && !strategy.equals("uniform")
            && !strategy.equals("constant")) {
            throw new IllegalArgumentException("Unsupported strategy: " + strategy);
        }
        this.strategy = strategy;
        this.constant = constant;
    }

    @Override
    public DummyClassifier fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        this.nFeatures = X.cols();

        Set<Integer> unique = new LinkedHashSet<>();
        for (int i = 0; i < y.size(); i++) unique.add((int) y.get(i));
        classes = unique.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(classes);

        classPrior = new double[classes.length];
        for (int i = 0; i < y.size(); i++) {
            for (int c = 0; c < classes.length; c++) {
                if ((int) y.get(i) == classes[c]) {
                    classPrior[c]++;
                    break;
                }
            }
        }
        for (int c = 0; c < classes.length; c++) {
            classPrior[c] /= y.size();
        }

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "DummyClassifier");
        Random rng = new Random(42);
        Vector result = new Vector(X.rows());
        for (int i = 0; i < X.rows(); i++) {
            switch (strategy) {
                case "prior":
                case "most_frequent":
                    int bestIdx = 0;
                    for (int c = 1; c < classes.length; c++) {
                        if (classPrior[c] > classPrior[bestIdx]) bestIdx = c;
                    }
                    result.set(i, classes[bestIdx]);
                    break;
                case "stratified":
                    double r = rng.nextDouble();
                    double cum = 0;
                    int sel = 0;
                    for (int c = 0; c < classes.length; c++) {
                        cum += classPrior[c];
                        if (r < cum) { sel = c; break; }
                    }
                    result.set(i, classes[sel]);
                    break;
                case "uniform":
                    result.set(i, classes[rng.nextInt(classes.length)]);
                    break;
                case "constant":
                    result.set(i, constant != null ? constant : 0);
                    break;
            }
        }
        return result;
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "DummyClassifier");
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
        p.put("strategy", strategy);
        p.put("constant", constant);
        return Collections.unmodifiableMap(p);
    }
}
