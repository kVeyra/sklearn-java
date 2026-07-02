package org.sklearn.neighbors;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Radius-based nearest-neighbors classifier.
 */
public class RadiusNeighborsClassifier implements Predictor<Matrix, Vector, Vector> {

    private double radius;
    private boolean outlierLabelEnabled;
    private double outlierLabelValue;
    private boolean fitted;
    private Matrix xTrain;
    private Vector yTrain;
    private int[] classes;
    private int nFeatures;
    private boolean weights;
    private String weightsStr;

    public RadiusNeighborsClassifier() {
        this(1.0, "uniform", false, -1.0);
    }

    public RadiusNeighborsClassifier(double radius, String weights,
                                     boolean outlierLabelEnabled, double outlierLabelValue) {
        this.radius = radius;
        this.weightsStr = weights;
        this.weights = "distance".equals(weights);
        this.outlierLabelEnabled = outlierLabelEnabled;
        this.outlierLabelValue = outlierLabelValue;
    }

    @Override
    public RadiusNeighborsClassifier fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        this.xTrain = new Matrix(X);
        this.yTrain = new Vector(y);
        this.nFeatures = X.cols();

        Set<Integer> labels = new LinkedHashSet<>();
        for (int i = 0; i < y.size(); i++) {
            labels.add((int) y.get(i));
        }
        this.classes = labels.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(this.classes);
        this.fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "RadiusNeighborsClassifier");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures + " got " + X.cols());
        }

        int n = X.rows();
        int nTrain = xTrain.rows();
        double[] preds = new double[n];

        for (int i = 0; i < n; i++) {
            List<Integer> neighborIdx = new ArrayList<>();
            List<Double> neighborDists = new ArrayList<>();
            for (int j = 0; j < nTrain; j++) {
                double dist = 0.0;
                for (int f = 0; f < nFeatures; f++) {
                    double diff = X.get(i, f) - xTrain.get(j, f);
                    dist += diff * diff;
                }
                dist = Math.sqrt(dist);
                if (dist <= radius) {
                    neighborIdx.add(j);
                    neighborDists.add(dist);
                }
            }

            if (neighborIdx.isEmpty()) {
                if (outlierLabelEnabled) {
                    preds[i] = outlierLabelValue;
                } else {
                    preds[i] = -1;
                }
                continue;
            }

            Map<Integer, Double> voteSum = new HashMap<>();
            for (int ni = 0; ni < neighborIdx.size(); ni++) {
                int label = (int) yTrain.get(neighborIdx.get(ni));
                double w = weights ? 1.0 / Math.max(neighborDists.get(ni), 1e-10) : 1.0;
                voteSum.merge(label, w, Double::sum);
            }

            int bestLabel = 0;
            double bestWeight = -1;
            for (var entry : voteSum.entrySet()) {
                if (entry.getValue() > bestWeight) {
                    bestWeight = entry.getValue();
                    bestLabel = entry.getKey();
                }
            }
            preds[i] = bestLabel;
        }
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Vector pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y.size(); i++) {
            if (Math.abs(pred.get(i) - y.get(i)) < 0.5) {
                correct++;
            }
        }
        return (double) correct / y.size();
    }

    public int[] getClasses() {
        return classes;
    }
    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("radius", radius);
        p.put("weights", weightsStr);
        p.put("outlier_label_enabled", outlierLabelEnabled);
        p.put("outlier_label_value", outlierLabelValue);
        return Collections.unmodifiableMap(p);
    }
}
