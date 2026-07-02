package org.sklearn.preprocessing;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Binarize labels in a one-vs-all fashion.
 *
 * <p>Mirrors {@code sklearn.preprocessing.LabelBinarizer}.
 *
 * <p>Usage:
 * <pre>{@code
 * LabelBinarizer lb = new LabelBinarizer();
 * lb.fit(new Vector(new double[]{1, 2, 3}));
 * Matrix binarized = lb.transform(new Vector(new double[]{1, 3}));
 * }</pre>
 */
public class LabelBinarizer {

    private boolean fitted;
    private int[] classes;
    private int nClasses;

    /**
     * Fit label binarizer.
     */
    public LabelBinarizer fit(Vector y) {
        Validation.checkTarget(y, -1);
        Set<Integer> seen = new LinkedHashSet<>();
        for (int i = 0; i < y.size(); i++) {
            seen.add((int) y.get(i));
        }
        classes = seen.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(classes);
        nClasses = classes.length;
        fitted = true;
        return this;
    }

    /**
     * Transform labels to binarized form.
     */
    public Matrix transform(Vector y) {
        Validation.checkFitted(fitted, "LabelBinarizer");
        Validation.checkTarget(y, -1);
        Matrix result = new Matrix(y.size(), nClasses);
        for (int i = 0; i < y.size(); i++) {
            int label = (int) y.get(i);
            int idx = Arrays.binarySearch(classes, label);
            if (idx < 0) {
                throw new IllegalArgumentException("Label " + label + " not seen during fit");
            }
            result.set(i, idx, 1.0);
        }
        return result;
    }

    /**
     * Inverse transform: convert binarized matrix back to label vector.
     */
    public Vector inverseTransform(Matrix Y) {
        Validation.checkFitted(fitted, "LabelBinarizer");
        Validation.checkMatrix(Y, -1);
        double[] labels = new double[Y.rows()];
        for (int i = 0; i < Y.rows(); i++) {
            int maxIdx = 0;
            for (int j = 1; j < Y.cols(); j++) {
                if (Y.get(i, j) > Y.get(i, maxIdx)) {
                    maxIdx = j;
                }
            }
            labels[i] = classes[maxIdx];
        }
        return new Vector(labels);
    }

    public int[] getClasses() {
        return classes;
    }

    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("classes", classes);
        return params;
    }
}
