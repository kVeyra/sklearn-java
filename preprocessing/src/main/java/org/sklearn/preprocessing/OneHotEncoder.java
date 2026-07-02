package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Encode categorical integer features as a one-hot numeric array.
 *
 * <p>This is the Java equivalent of {@code sklearn.preprocessing.OneHotEncoder}.
 */
public class OneHotEncoder implements Transformer<Matrix, Void> {

    /**
     * Strategy for dropping a category per feature.
     */
    public enum Drop {
        NONE, FIRST
    }

    private List<int[]> categories;

    private boolean fitted;

    private Drop drop;

    public OneHotEncoder() {
        this.drop = Drop.NONE;
    }

    public OneHotEncoder(boolean dropFirst) {
        this.drop = dropFirst ? Drop.FIRST : Drop.NONE;
    }

    @Override
    public OneHotEncoder fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int nFeatures = X.cols();
        int nSamples = X.rows();

        List<int[]> cats = new ArrayList<>(nFeatures);
        for (int j = 0; j < nFeatures; j++) {
            int[] col = new int[nSamples];
            for (int i = 0; i < nSamples; i++) {
                col[i] = (int) X.get(i, j);
            }
            cats.add(Arrays.stream(col).distinct().sorted().toArray());
        }

        this.categories = cats;
        fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "OneHotEncoder");
        Validation.checkMatrix(X, -1);
        if (X.cols() != categories.size()) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + categories.size()
                    + " features, got " + X.cols());
        }

        int nSamples = X.rows();
        int nFeatures = categories.size();

        int[] offsets = new int[nFeatures];
        int totalCols = 0;
        for (int j = 0; j < nFeatures; j++) {
            offsets[j] = totalCols;
            totalCols += categories.get(j).length - (drop == Drop.FIRST ? 1 : 0);
        }

        double[][] result = new double[nSamples][totalCols];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                int val = (int) X.get(i, j);
                int[] cats = this.categories.get(j);
                int idx = Arrays.binarySearch(cats, val);
                if (idx < 0) {
                    throw new IllegalArgumentException(
                        "Value " + val + " in feature " + j
                            + " was not seen during fit. Known categories: "
                            + Arrays.toString(cats));
                }
                if (drop == Drop.FIRST) {
                    if (idx == 0) {
                        continue;
                    }
                    idx--;
                }
                result[i][offsets[j] + idx] = 1.0;
            }
        }

        return new Matrix(result);
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        Validation.checkFitted(fitted, "OneHotEncoder");
        Validation.checkMatrix(X, -1);

        int nSamples = X.rows();
        int nFeatures = categories.size();

        int[] blockSizes = new int[nFeatures];
        int totalCols = 0;
        for (int j = 0; j < nFeatures; j++) {
            blockSizes[j] = categories.get(j).length - (drop == Drop.FIRST ? 1 : 0);
            totalCols += blockSizes[j];
        }

        if (X.cols() != totalCols) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + totalCols
                    + " columns, got " + X.cols());
        }

        double[][] result = new double[nSamples][nFeatures];
        for (int i = 0; i < nSamples; i++) {
            int offset = 0;
            for (int j = 0; j < nFeatures; j++) {
                int blockLen = blockSizes[j];
                int oneIdx = -1;
                for (int k = 0; k < blockLen; k++) {
                    if (X.get(i, offset + k) == 1.0) {
                        oneIdx = k;
                        break;
                    }
                }
                int[] cats = categories.get(j);
                if (oneIdx < 0) {
                    if (drop == Drop.FIRST) {
                        result[i][j] = cats[0];
                    } else {
                        throw new IllegalArgumentException(
                            "Invalid one-hot vector at sample " + i + ", feature " + j
                                + ": no 1 found in block");
                    }
                } else {
                    result[i][j] = cats[drop == Drop.FIRST ? oneIdx + 1 : oneIdx];
                }
                offset += blockLen;
            }
        }

        return new Matrix(result);
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
