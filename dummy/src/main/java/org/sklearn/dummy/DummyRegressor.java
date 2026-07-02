package org.sklearn.dummy;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

public class DummyRegressor implements Predictor<Matrix, Vector, Vector> {

    private String strategy;
    private Double constant;
    private boolean fitted;
    private double predictedValue;
    private int nFeatures;

    public DummyRegressor() {
        this("mean", null);
    }

    public DummyRegressor(String strategy, Double constant) {
        if (!strategy.equals("mean") && !strategy.equals("median")
            && !strategy.equals("quantile") && !strategy.equals("constant")) {
            throw new IllegalArgumentException("Unsupported strategy: " + strategy);
        }
        this.strategy = strategy;
        this.constant = constant;
    }

    @Override
    public DummyRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        this.nFeatures = X.cols();

        switch (strategy) {
            case "mean":
                predictedValue = y.mean();
                break;
            case "median":
                double[] sorted = new double[y.size()];
                for (int i = 0; i < y.size(); i++) sorted[i] = y.get(i);
                Arrays.sort(sorted);
                int mid = sorted.length / 2;
                predictedValue = sorted.length % 2 == 0
                    ? (sorted[mid - 1] + sorted[mid]) / 2.0
                    : sorted[mid];
                break;
            case "quantile":
                predictedValue = constant != null ? constant : 0.5;
                break;
            case "constant":
                predictedValue = constant != null ? constant : 0.0;
                break;
        }

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "DummyRegressor");
        Vector result = new Vector(X.rows());
        for (int i = 0; i < X.rows(); i++) {
            result.set(i, predictedValue);
        }
        return result;
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "DummyRegressor");
        Vector preds = predict(X);
        double ssRes = 0, ssTot = 0;
        double yMean = y.mean();
        for (int i = 0; i < y.size(); i++) {
            double diff = y.get(i) - preds.get(i);
            ssRes += diff * diff;
            double diffTot = y.get(i) - yMean;
            ssTot += diffTot * diffTot;
        }
        return ssTot > 0 ? 1.0 - ssRes / ssTot : (ssRes == 0 ? 1.0 : 0.0);
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
