package org.sklearn.impute;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

public class SimpleImputer implements Transformer<Matrix, Void> {

    private String strategy;
    private double fillValue;
    private boolean fitted;
    private double[] statistics;
    private int nFeatures;

    public SimpleImputer() {
        this("mean", 0.0);
    }

    public SimpleImputer(String strategy, double fillValue) {
        if (!strategy.equals("mean") && !strategy.equals("median")
            && !strategy.equals("most_frequent") && !strategy.equals("constant")) {
            throw new IllegalArgumentException(
                "Unsupported strategy: " + strategy + ". Use mean, median, most_frequent, or constant.");
        }
        this.strategy = strategy;
        this.fillValue = fillValue;
    }

    @Override
    public SimpleImputer fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;
        this.statistics = new double[m];

        for (int j = 0; j < m; j++) {
            int count = 0;
            double sum = 0;
            List<Double> values = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                double v = X.get(i, j);
                if (!Double.isNaN(v)) {
                    sum += v;
                    count++;
                    values.add(v);
                }
            }
            switch (strategy) {
                case "mean":
                    statistics[j] = count > 0 ? sum / count : 0;
                    break;
                case "median":
                    if (values.isEmpty()) {
                        statistics[j] = 0;
                    } else {
                        double[] arr = values.stream().mapToDouble(Double::doubleValue).toArray();
                        Arrays.sort(arr);
                        int mid = arr.length / 2;
                        statistics[j] = arr.length % 2 == 0
                            ? (arr[mid - 1] + arr[mid]) / 2.0
                            : arr[mid];
                    }
                    break;
                case "most_frequent":
                    if (values.isEmpty()) {
                        statistics[j] = 0;
                    } else {
                        Map<Double, Integer> freq = new HashMap<>();
                        for (double v : values) {
                            freq.put(v, freq.getOrDefault(v, 0) + 1);
                        }
                        double bestVal = values.get(0);
                        int bestCount = 0;
                        for (Map.Entry<Double, Integer> e : freq.entrySet()) {
                            if (e.getValue() > bestCount) {
                                bestCount = e.getValue();
                                bestVal = e.getKey();
                            }
                        }
                        statistics[j] = bestVal;
                    }
                    break;
                case "constant":
                    statistics[j] = fillValue;
                    break;
            }
        }
        this.fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "SimpleImputer");
        Validation.checkMatrix(X, nFeatures);
        Matrix result = new Matrix(X.rows(), X.cols());
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                double v = X.get(i, j);
                result.set(i, j, Double.isNaN(v) ? statistics[j] : v);
            }
        }
        return result;
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        throw new UnsupportedOperationException("SimpleImputer does not support inverseTransform");
    }

    public boolean isFitted() {
        return fitted;
    }

    public double[] getStatistics() {
        return statistics;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("strategy", strategy);
        p.put("fill_value", fillValue);
        return Collections.unmodifiableMap(p);
    }
}
