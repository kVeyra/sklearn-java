package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

class FunctionTransformerTest {

    @Test
    void testElementWiseFunction() {
        Function<Double, Double> square = x -> x * x;
        FunctionTransformer ft = new FunctionTransformer(square, true);
        ft.fit(new Matrix(new double[][]{{1.0, 2.0}}), null);
        Matrix X = new Matrix(new double[][]{{2.0, 3.0}});
        Matrix Xt = ft.transform(X);
        assertEquals(4.0, Xt.get(0, 0), 1e-10);
        assertEquals(9.0, Xt.get(0, 1), 1e-10);
    }

    @Test
    void testMatrixWiseFunction() {
        Function<Matrix, Matrix> transpose = m -> {
            int r = m.rows(), c = m.cols();
            double[][] t = new double[c][r];
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    t[j][i] = m.get(i, j);
                }
            }
            return new Matrix(t);
        };
        FunctionTransformer ft = new FunctionTransformer(transpose);
        Matrix X = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        Matrix Xt = ft.transform(X);
        assertEquals(2, Xt.rows());
        assertEquals(2, Xt.cols());
        assertEquals(1.0, Xt.get(0, 0), 1e-10);
        assertEquals(3.0, Xt.get(0, 1), 1e-10);
    }

    @Test
    void testInverseThrows() {
        Function<Double, Double> f = x -> x;
        FunctionTransformer ft = new FunctionTransformer(f, true);
        ft.fit(new Matrix(new double[][]{{1.0}}), null);
        assertThrows(UnsupportedOperationException.class,
            () -> ft.inverseTransform(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testGetParameters() {
        Function<Double, Double> f = x -> x;
        FunctionTransformer ft = new FunctionTransformer(f, true);
        var params = ft.getParameters();
        assertTrue((boolean) params.get("has_element_func"));
    }
}
