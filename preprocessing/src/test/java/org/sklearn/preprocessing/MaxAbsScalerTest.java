package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class MaxAbsScalerTest {

    @Test
    void testFitAndTransform() {
        Matrix X = new Matrix(new double[][]{{-2.0}, {1.0}, {3.0}});
        MaxAbsScaler mas = new MaxAbsScaler();
        mas.fit(X, null);
        Matrix Xt = mas.transform(X);
        assertEquals(1.0, Xt.get(2, 0), 1e-10);
        assertEquals(-2.0 / 3.0, Xt.get(0, 0), 1e-10);
    }

    @Test
    void testInverseTransform() {
        Matrix X = new Matrix(new double[][]{{-2.0, 4.0}, {1.0, -8.0}});
        MaxAbsScaler mas = new MaxAbsScaler();
        mas.fit(X, null);
        Matrix Xt = mas.transform(X);
        Matrix Xback = mas.inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                assertEquals(X.get(i, j), Xback.get(i, j), 1e-10);
            }
        }
    }

    @Test
    void testTransformBeforeFitThrows() {
        MaxAbsScaler mas = new MaxAbsScaler();
        assertThrows(IllegalStateException.class,
            () -> mas.transform(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testIsFitted() {
        MaxAbsScaler mas = new MaxAbsScaler();
        assertFalse(mas.isFitted());
        mas.fit(new Matrix(new double[][]{{1.0}, {2.0}}), null);
        assertTrue(mas.isFitted());
    }
}
