package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Generate polynomial and interaction features.
 *
 * <p>Generates a feature matrix consisting of all polynomial combinations
 * of the features with degree less than or equal to the specified degree.
 * For example, if input has two features [a, b] and degree=2, the output
 * will be [1, a, b, a^2, a*b, b^2].
 *
 * <p>Mirrors {@code sklearn.preprocessing.PolynomialFeatures}.
 */
public class PolynomialFeatures implements Transformer<Matrix, Void> {

    private int degree;
    private boolean includeBias;
    private boolean interactionOnly;
    private int nFeaturesIn;
    private int[][] combinations;
    private boolean fitted;

    /**
     * Create PolynomialFeatures with given degree and bias.
     *
     * @param degree      polynomial degree (>= 1)
     * @param includeBias if true, include a bias column (all ones) as first column
     */
    public PolynomialFeatures(int degree, boolean includeBias) {
        this(degree, includeBias, false);
    }

    /**
     * Create PolynomialFeatures with full control.
     *
     * @param degree          polynomial degree
     * @param includeBias     if true, include bias column
     * @param interactionOnly if true, only include interaction features (no powers)
     */
    public PolynomialFeatures(int degree, boolean includeBias, boolean interactionOnly) {
        if (degree < 1) {
            throw new IllegalArgumentException("degree must be >= 1");
        }
        this.degree = degree;
        this.includeBias = includeBias;
        this.interactionOnly = interactionOnly;
    }

    @Override
    public PolynomialFeatures fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        nFeaturesIn = X.cols();
        List<int[]> combos = new ArrayList<>();

        if (includeBias) {
            combos.add(new int[0]);
        }

        for (int d = 1; d <= degree; d++) {
            generateCombinations(combos, nFeaturesIn, d, 0, new int[d], 0);
        }

        combinations = combos.toArray(new int[0][]);
        fitted = true;
        return this;
    }

    private void generateCombinations(List<int[]> result, int n, int targetDegree,
                                       int start, int[] current, int pos) {
        if (pos == targetDegree) {
            result.add(current.clone());
            return;
        }
        for (int i = start; i < n; i++) {
            current[pos] = i;
            generateCombinations(result, n, targetDegree,
                interactionOnly ? i + 1 : i, current, pos + 1);
        }
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "PolynomialFeatures");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeaturesIn) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeaturesIn
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        int nOut = combinations.length;
        double[][] result = new double[n][nOut];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nOut; j++) {
                double val = 1.0;
                for (int k = 0; k < combinations[j].length; k++) {
                    val *= X.get(i, combinations[j][k]);
                }
                result[i][j] = val;
            }
        }
        return new Matrix(result);
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        throw new UnsupportedOperationException(
            "PolynomialFeatures does not support inverse_transform");
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("degree", degree);
        params.put("include_bias", includeBias);
        params.put("interaction_only", interactionOnly);
        return Collections.unmodifiableMap(params);
    }

    public boolean isFitted() {
        return fitted;
    }

    /**
     * Return the number of output features after transformation.
     */
    public int getOutputDimension() {
        Validation.checkFitted(fitted, "PolynomialFeatures");
        return combinations.length;
    }
}
