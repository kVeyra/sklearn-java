package org.sklearn.feature_selection;

import org.sklearn.core.Transformer;
import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Feature ranking with recursive feature elimination.
 *
 * <p>Mirrors {@code sklearn.feature_selection.RFE}.
 *
 * <p>Usage:
 * <pre>{@code
 * RFE selector = new RFE(estimator, 5);
 * selector.fit(X, y);
 * Matrix Xreduced = selector.transform(X);
 * }</pre>
 */
public class RFE implements Transformer<Matrix, Matrix> {

    private Predictor<Matrix, Vector, Vector> estimator;
    private int nFeaturesToSelect;
    private int step;
    private boolean fitted;

    private int[] selectedIndices;
    private int[] ranking;
    private double[] support;
    private int nFeatures;

    public RFE(Predictor<Matrix, Vector, Vector> estimator, int nFeaturesToSelect) {
        this(estimator, nFeaturesToSelect, 1);
    }

    public RFE(Predictor<Matrix, Vector, Vector> estimator, int nFeaturesToSelect, int step) {
        this.estimator = estimator;
        this.nFeaturesToSelect = nFeaturesToSelect;
        this.step = step;
    }

    @Override
    public RFE fit(Matrix X, Matrix y) {
        return fit(X, y.col(0));
    }

    public RFE fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        boolean[] remaining = new boolean[m];
        Arrays.fill(remaining, true);
        ranking = new int[m];
        Arrays.fill(ranking, 1);

        int nRemaining = m;
        int rank = 1;

        while (nRemaining > nFeaturesToSelect) {
            int nRemove = Math.min(step, nRemaining - nFeaturesToSelect);
            nRemaining -= nRemove;

            Matrix xSub = selectFeatures(X, remaining);
            estimator.fit(xSub, y);

            double[] coefs = getImportance(estimator, nRemaining + nRemove);

            List<Integer> candidates = new ArrayList<>();
            for (int j = 0; j < m; j++) {
                if (remaining[j]) {
                    candidates.add(j);
                }
            }

            Integer[] sorted = new Integer[candidates.size()];
            for (int i = 0; i < sorted.length; i++) {
                sorted[i] = i;
            }
            int finalNRemove = nRemove;
            Arrays.sort(sorted, (a, b) -> Double.compare(
                Math.abs(coefs[a]), Math.abs(coefs[b])));

            for (int i = 0; i < sorted.length; i++) {
                int origIdx = candidates.get(sorted[i]);
                if (i < finalNRemove) {
                    ranking[origIdx] = rank;
                    remaining[origIdx] = false;
                }
            }

            rank++;
        }

        List<Integer> selected = new ArrayList<>();
        for (int j = 0; j < m; j++) {
            if (remaining[j]) {
                selected.add(j);
            }
        }
        selectedIndices = selected.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(selectedIndices);

        support = new double[m];
        for (int j = 0; j < m; j++) {
            support[j] = remaining[j] ? 1.0 : 0.0;
        }

        this.fitted = true;
        return this;
    }

    private Matrix selectFeatures(Matrix X, boolean[] remaining) {
        int count = 0;
        for (boolean r : remaining) {
            if (r) {
                count++;
            }
        }
        Matrix result = new Matrix(X.rows(), count);
        int col = 0;
        for (int j = 0; j < remaining.length; j++) {
            if (remaining[j]) {
                for (int i = 0; i < X.rows(); i++) {
                    result.set(i, col, X.get(i, j));
                }
                col++;
            }
        }
        return result;
    }

    private double[] getImportance(Predictor<Matrix, Vector, Vector> est, int nFeats) {
        Map<String, Object> params = est.getParameters();
        if (params.containsKey("coef_")) {
            Object c = params.get("coef_");
            if (c instanceof Vector) {
                Vector v = (Vector) c;
                double[] res = new double[v.size()];
                for (int i = 0; i < v.size(); i++) {
                    res[i] = v.get(i);
                }
                if (res.length == 1 && nFeats > 1) {
                    return null;
                }
                return res;
            } else if (c instanceof Matrix) {
                Matrix m = (Matrix) c;
                if (m.cols() == nFeats) {
                    double[] res = new double[nFeats];
                    for (int j = 0; j < nFeats; j++) {
                        double sum = 0;
                        for (int r = 0; r < m.rows(); r++) {
                            sum += Math.abs(m.get(r, j));
                        }
                        res[j] = sum;
                    }
                    return res;
                }
            }
        }
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
        double[] res = new double[nFeats];
        Arrays.fill(res, 1.0);
        return res;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "RFE");
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
        throw new UnsupportedOperationException("RFE does not support inverseTransform");
    }

    @Override
    public Matrix fitTransform(Matrix X, Matrix y) {
        return fit(X, y).transform(X);
    }

    public Matrix fitTransform(Matrix X, Vector y) {
        return fit(X, y).transform(X);
    }

    public int[] getSupport() {
        return getSelectedIndices();
    }

    public int[] getSelectedIndices() {
        return selectedIndices;
    }

    public int[] getRanking() {
        return ranking;
    }

    public double[] getSupportArray() {
        return support;
    }

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_features_to_select", nFeaturesToSelect);
        p.put("step", step);
        p.put("estimator", estimator);
        return Collections.unmodifiableMap(p);
    }
}
