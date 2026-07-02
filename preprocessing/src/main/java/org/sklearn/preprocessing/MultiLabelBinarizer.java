package org.sklearn.preprocessing;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.*;

/**
 * Transform between iterable of iterables and a multilabel format.
 *
 * <p>Mirrors {@code sklearn.preprocessing.MultiLabelBinarizer}.
 *
 * <p>Usage:
 * <pre>{@code
 * MultiLabelBinarizer mlb = new MultiLabelBinarizer();
 * mlb.fit(new int[][]{{1, 2}, {3}});
 * Matrix binarized = mlb.transform(new int[][]{{1, 3}});
 * }</pre>
 */
public class MultiLabelBinarizer {

    private boolean fitted;
    private int[] classes;
    private int nClasses;

    /**
     * Fit given label sets.
     */
    public MultiLabelBinarizer fit(int[][] y) {
        Set<Integer> seen = new LinkedHashSet<>();
        for (int[] row : y) {
            for (int v : row) {
                seen.add(v);
            }
        }
        classes = seen.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(classes);
        nClasses = classes.length;
        fitted = true;
        return this;
    }

    /**
     * Transform multilabel indicator to binarized matrix.
     */
    public Matrix transform(int[][] y) {
        if (!fitted) {
            throw new IllegalStateException("MultiLabelBinarizer is not fitted");
        }
        Matrix result = new Matrix(y.length, nClasses);
        for (int i = 0; i < y.length; i++) {
            for (int v : y[i]) {
                int idx = Arrays.binarySearch(classes, v);
                if (idx >= 0) {
                    result.set(i, idx, 1.0);
                }
            }
        }
        return result;
    }

    /**
     * Inverse transform: convert binarized matrix back to label sets.
     */
    public int[][] inverseTransform(Matrix Y) {
        if (!fitted) {
            throw new IllegalStateException("MultiLabelBinarizer is not fitted");
        }
        int[][] result = new int[Y.rows()][];
        for (int i = 0; i < Y.rows(); i++) {
            List<Integer> labels = new ArrayList<>();
            for (int j = 0; j < Y.cols(); j++) {
                if (Y.get(i, j) > 0.5) {
                    labels.add(classes[j]);
                }
            }
            result[i] = labels.stream().mapToInt(Integer::intValue).toArray();
        }
        return result;
    }

    public int[] getClasses() {
        return classes;
    }
}
