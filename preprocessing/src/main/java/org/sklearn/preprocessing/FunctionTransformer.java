package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.utils.Validation;

import java.util.*;
import java.util.function.Function;

/**
 * Constructs a transformer from an arbitrary function.
 *
 * <p>This transformer applies a user-supplied function to the input data.
 * The function is applied element-wise by default, or over the full matrix
 * if the function takes a {@link Matrix} argument.
 *
 * <p>Mirrors {@code sklearn.preprocessing.FunctionTransformer}.
 */
public class FunctionTransformer implements Transformer<Matrix, Void> {

    private Function<Double, Double> elementFunc;
    private Function<Matrix, Matrix> matrixFunc;
    private boolean validate;

    /**
     * Create a FunctionTransformer with an element-wise function.
     *
     * @param func     function that maps a single double value to a new double value
     * @param validate if true, validate input data dimensions
     */
    public FunctionTransformer(Function<Double, Double> func, boolean validate) {
        this.elementFunc = func;
        this.validate = validate;
    }

    /**
     * Create a FunctionTransformer with a matrix-level function.
     *
     * @param func     function that maps a Matrix to a new Matrix
     * @param validate if true, validate input data dimensions
     */
    public FunctionTransformer(Function<Matrix, Matrix> func) {
        this.matrixFunc = func;
        this.validate = true;
    }

    @Override
    public FunctionTransformer fit(Matrix X, Void y) {
        if (validate) {
            Validation.checkMatrix(X, -1);
        }
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        if (validate) {
            Validation.checkMatrix(X, -1);
        }

        if (matrixFunc != null) {
            return matrixFunc.apply(X);
        }

        int n = X.rows(), m = X.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = elementFunc.apply(X.get(i, j));
            }
        }
        return new Matrix(result);
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        throw new UnsupportedOperationException(
            "FunctionTransformer does not support inverse_transform by default");
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("validate", validate);
        params.put("has_element_func", elementFunc != null);
        params.put("has_matrix_func", matrixFunc != null);
        return Collections.unmodifiableMap(params);
    }
}
