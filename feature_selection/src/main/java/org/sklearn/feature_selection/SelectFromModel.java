package org.sklearn.feature_selection;

import org.sklearn.core.Transformer;
import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Feature selector that selects features based on importance weights from a fitted estimator.
 *
 * <p>Mirrors {@code sklearn.feature_selection.SelectFromModel}.
 *
 * <p>Usage:
 * <pre>{@code
 * SelectFromModel selector = new SelectFromModel(estimator);
 * selector.fit(X, y);
 * Matrix Xreduced = selector.transform(X);
 * }</pre>
 */
public class SelectFromModel implements Transformer<Matrix, Matrix> {

    private Predictor<Matrix, Vector, Vector> estimator;
    private double threshold;
    private String thresholdType;
    private boolean fitted;

    private int[] selectedIndices;
    private int nFeatures;

    public SelectFromModel(Predictor<Matrix, Vector, Vector> estimator) {
        this(estimator, null, "mean");
    }

    public SelectFromModel(Predictor<Matrix, Vector, Vector> estimator, Double threshold, String thresholdType) {
        this.estimator = estimator;
        this.threshold = threshold != null ? threshold : Double.MAX_VALUE;
        this.thresholdType = thresholdType;
    }

    @Override
    public SelectFromModel fit(Matrix X, Matrix y) {
        return fit(X, y.col(0));
    }

    public SelectFromModel fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        this.nFeatures = X.cols();

        estimator.fit(X, y);
        double[] importances = getFeatureImportances();

        double actualThreshold;
        if (threshold != Double.MAX_VALUE) {
            actualThreshold = threshold;
        } else {
            switch (thresholdType) {
                case "mean":
                    double sum = 0;
                    for (double v : importances) {
                        sum += v;
                    }
                    actualThreshold = sum / importances.length;
                    break;
                case "median":
                    double[] sorted = importances.clone();
                    Arrays.sort(sorted);
                    actualThreshold = sorted[sorted.length / 2];
                    break;
                default:
                    actualThreshold = 0;
            }
        }

        List<Integer> selected = new ArrayList<>();
        for (int j = 0; j < nFeatures; j++) {
            if (importances[j] >= actualThreshold && importances[j] > 0) {
                selected.add(j);
            }
        }

        // Fallback: select at least 1 feature
        if (selected.isEmpty()) {
            int bestIdx = 0;
            for (int j = 1; j < nFeatures; j++) {
                if (importances[j] > importances[bestIdx]) {
                    bestIdx = j;
                }
            }
            selected.add(bestIdx);
        }

        selectedIndices = selected.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(selectedIndices);

        this.fitted = true;
        return this;
    }

    private double[] getFeatureImportances() {
        Map<String, Object> params = estimator.getParameters();
        if (params.containsKey("feature_importances_")) {
            Object fi = params.get("feature_importances_");
            if (fi instanceof double[]) {
                return (double[]) fi;
            }
            if (fi instanceof Vector) {
                Vector v = (Vector) fi;
                double[] res = new double[v.size()];
                for (int i = 0; i < v.size(); i++) {
                    res[i] = v.get(i);
                }
                return res;
            }
        }
        if (params.containsKey("coef_")) {
            Object c = params.get("coef_");
            if (c instanceof Vector) {
                Vector v = (Vector) c;
                double[] res = new double[v.size()];
                for (int i = 0; i < v.size(); i++) {
                    res[i] = Math.abs(v.get(i));
                }
                return res;
            } else if (c instanceof Matrix) {
                Matrix m = (Matrix) c;
                double[] res = new double[nFeatures];
                if (m.cols() == nFeatures) {
                    for (int j = 0; j < nFeatures; j++) {
                        double sum = 0;
                        for (int r = 0; r < m.rows(); r++) {
                            sum += Math.abs(m.get(r, j));
                        }
                        res[j] = sum;
                    }
                }
                return res;
            }
        }
        double[] res = new double[nFeatures];
        Arrays.fill(res, 1.0);
        return res;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "SelectFromModel");
        Matrix result = new Matrix(X.rows(), selectedIndices.length);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < selectedIndices.length; j++) {
                result.set(i, j, X.get(i, selectedIndices[j]));
            }
        }
        return result;
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        throw new UnsupportedOperationException("SelectFromModel does not support inverseTransform");
    }

    @Override
    public Matrix fitTransform(Matrix X, Matrix y) {
        return fit(X, y).transform(X);
    }

    public Matrix fitTransform(Matrix X, Vector y) {
        return fit(X, y).transform(X);
    }

    public int[] getSupport() {
        return selectedIndices;
    }

    public int[] getSelectedIndices() {
        return selectedIndices;
    }

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("threshold", threshold != Double.MAX_VALUE ? threshold : null);
        p.put("threshold_type", thresholdType);
        p.put("estimator", estimator);
        return Collections.unmodifiableMap(p);
    }
}
