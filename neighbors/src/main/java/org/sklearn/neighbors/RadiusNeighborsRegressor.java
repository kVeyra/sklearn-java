package org.sklearn.neighbors;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Radius-based nearest-neighbors regressor.
 */
public class RadiusNeighborsRegressor implements Predictor<Matrix, Vector, Vector> {

    private double radius;
    private boolean weights;
    private String weightsStr;
    private boolean fitted;
    private Matrix xTrain;
    private Vector yTrain;
    private int nFeatures;

    public RadiusNeighborsRegressor() {
        this(1.0, "uniform");
    }

    public RadiusNeighborsRegressor(double radius, String weights) {
        this.radius = radius;
        this.weightsStr = weights;
        this.weights = "distance".equals(weights);
    }

    @Override
    public RadiusNeighborsRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        this.xTrain = new Matrix(X);
        this.yTrain = new Vector(y);
        this.nFeatures = X.cols();
        this.fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "RadiusNeighborsRegressor");
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
                preds[i] = 0.0;
                continue;
            }

            double sum = 0.0;
            double weightSum = 0.0;
            for (int ni = 0; ni < neighborIdx.size(); ni++) {
                double yVal = yTrain.get(neighborIdx.get(ni));
                double w = weights ? 1.0 / Math.max(neighborDists.get(ni), 1e-10) : 1.0;
                sum += w * yVal;
                weightSum += w;
            }
            preds[i] = sum / weightSum;
        }
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Vector pred = predict(X);
        double ssRes = 0.0, ssTot = 0.0, yMean = y.mean();
        for (int i = 0; i < y.size(); i++) {
            double diff = y.get(i) - pred.get(i);
            ssRes += diff * diff;
            double diffMean = y.get(i) - yMean;
            ssTot += diffMean * diffMean;
        }
        if (ssTot == 0.0) {
            return 1.0;
        }
        return 1.0 - ssRes / ssTot;
    }

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("radius", radius);
        p.put("weights", weightsStr);
        return Collections.unmodifiableMap(p);
    }
}
