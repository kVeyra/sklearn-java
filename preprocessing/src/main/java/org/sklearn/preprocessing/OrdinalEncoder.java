package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Encode categorical features as integer arrays (0..n_categories-1 per feature).
 *
 * <p>Each feature column is independently mapped to integers 0 through
 * n_categories-1, based on the unique values seen during fit.
 * Unknown categories during transform are mapped to -1.
 *
 * <p>Mirrors {@code sklearn.preprocessing.OrdinalEncoder}.
 */
public class OrdinalEncoder implements Transformer<Matrix, Void> {

    private List<String[]> categories;
    private Map<String, Integer>[] catMaps;
    private boolean fitted;

    @Override
    @SuppressWarnings("unchecked")
    public OrdinalEncoder fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int m = X.cols();
        categories = new ArrayList<>();
        catMaps = new LinkedHashMap[m];

        for (int j = 0; j < m; j++) {
            Set<String> unique = new LinkedHashSet<>();
            for (int i = 0; i < X.rows(); i++) {
                unique.add(formatValue(X.get(i, j)));
            }
            String[] sortedCats = unique.toArray(new String[0]);
            Arrays.sort(sortedCats);
            categories.add(sortedCats);
            catMaps[j] = new LinkedHashMap<>();
            for (int k = 0; k < sortedCats.length; k++) {
                catMaps[j].put(sortedCats[k], k);
            }
        }

        fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "OrdinalEncoder");
        Validation.checkMatrix(X, -1);
        if (X.cols() != categories.size()) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + categories.size()
                    + " features, got " + X.cols());
        }

        int n = X.rows(), m = X.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                String val = formatValue(X.get(i, j));
                Integer code = catMaps[j].get(val);
                result[i][j] = code != null ? code : -1;
            }
        }
        return new Matrix(result);
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        Validation.checkFitted(fitted, "OrdinalEncoder");
        Validation.checkMatrix(X, -1);
        if (X.cols() != categories.size()) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + categories.size()
                    + " features, got " + X.cols());
        }

        int n = X.rows(), m = X.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                int code = (int) X.get(i, j);
                String[] cats = categories.get(j);
                if (code >= 0 && code < cats.length) {
                    result[i][j] = Double.parseDouble(cats[code]);
                } else {
                    result[i][j] = -1;
                }
            }
        }
        return new Matrix(result);
    }

    private static String formatValue(double v) {
        if (v == Math.floor(v) && !Double.isInfinite(v)) {
            return String.valueOf((long) v);
        }
        return String.valueOf(v);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("categories", categories);
        return Collections.unmodifiableMap(params);
    }

    public boolean isFitted() {
        return fitted;
    }
}
