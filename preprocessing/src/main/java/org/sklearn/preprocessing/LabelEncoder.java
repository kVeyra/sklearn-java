package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.utils.Validation;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Encode target labels with value between 0 and n_classes-1.
 *
 * <p>This transformer learns a sorted set of unique class labels from the
 * training data and maps original labels to 0-based indices. This is the
 * Java equivalent of {@code sklearn.preprocessing.LabelEncoder}.
 *
 * <p>All operations are deterministic: repeated calls with the same input
 * produce identical results.
 */
public class LabelEncoder implements Transformer<int[], Void> {

    private int[] classes;

    private boolean fitted;

    /**
     * Fit the label encoder by finding and sorting the unique classes in y.
     *
     * @param y        target labels to encode
     * @param ignored  ignored (may be {@code null})
     * @return this fitted encoder
     */
    @Override
    public LabelEncoder fit(int[] y, Void ignored) {
        Validation.checkLabels(y, -1);
        classes = Arrays.stream(y).distinct().sorted().toArray();
        fitted = true;
        return this;
    }

    /**
     * Transform labels to encoded indices.
     *
     * <p>Each input value is mapped to its 0-based index in the sorted
     * classes array learned during {@link #fit}.
     *
     * @param y  labels to encode
     * @return encoded labels, each in [0, n_classes)
     * @throws IllegalStateException    if the encoder is not fitted
     * @throws IllegalArgumentException if any label was not seen during fit
     */
    @Override
    public int[] transform(int[] y) {
        Validation.checkFitted(fitted, "LabelEncoder");
        Validation.checkLabels(y, -1);
        int[] encoded = new int[y.length];
        for (int i = 0; i < y.length; i++) {
            int idx = Arrays.binarySearch(classes, y[i]);
            if (idx < 0) {
                throw new IllegalArgumentException(
                    "Label " + y[i] + " was not seen during fit. Known classes: "
                        + Arrays.toString(classes));
            }
            encoded[i] = idx;
        }
        return encoded;
    }

    /**
     * Transform labels back to original encoding.
     *
     * <p>Each encoded index is mapped back to its original class value.
     *
     * @param y  encoded labels, each in [0, n_classes)
     * @return original labels
     * @throws IllegalStateException    if the encoder is not fitted
     * @throws IllegalArgumentException if any encoded index is out of range
     */
    @Override
    public int[] inverseTransform(int[] y) {
        Validation.checkFitted(fitted, "LabelEncoder");
        Validation.checkLabels(y, -1);
        int[] decoded = new int[y.length];
        for (int i = 0; i < y.length; i++) {
            if (y[i] < 0 || y[i] >= classes.length) {
                throw new IllegalArgumentException(
                    "Encoded label " + y[i] + " is out of range [0, " + classes.length + ")");
            }
            decoded[i] = classes[y[i]];
        }
        return decoded;
    }

    /**
     * Return the fitted parameters as an unmodifiable map.
     *
     * <p>The returned map contains the following key:
     * <ul>
     *   <li>{@code "classes"} &mdash; sorted unique class labels as an {@code int[]}</li>
     * </ul>
     *
     * @return unmodifiable map of parameter name to value
     */
    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("classes", classes);
        return Collections.unmodifiableMap(params);
    }

    /**
     * Check whether this encoder has been fitted to data.
     *
     * @return {@code true} if {@link #fit} has been called
     */
    public boolean isFitted() {
        return fitted;
    }
}
