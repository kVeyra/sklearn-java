package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class OrdinalEncoderTest {

    @Test
    void testRoundTrip() {
        Matrix X = new Matrix(new double[][]{
            {0.0, 10.0},
            {1.0, 20.0},
            {0.0, 20.0}
        });
        OrdinalEncoder oe = new OrdinalEncoder();
        oe.fit(X, null);
        Matrix Xt = oe.transform(X);
        assertEquals(0.0, Xt.get(0, 0), 1e-10);
        assertEquals(1.0, Xt.get(1, 0), 1e-10);
        assertEquals(0.0, Xt.get(2, 0), 1e-10);

        Matrix Xback = oe.inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                assertEquals(X.get(i, j), Xback.get(i, j), 1e-10);
            }
        }
    }

    @Test
    void testTransformBeforeFitThrows() {
        OrdinalEncoder oe = new OrdinalEncoder();
        assertThrows(IllegalStateException.class,
            () -> oe.transform(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        OrdinalEncoder oe = new OrdinalEncoder();
        oe.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}), null);
        assertThrows(IllegalArgumentException.class,
            () -> oe.transform(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testIsFitted() {
        OrdinalEncoder oe = new OrdinalEncoder();
        assertFalse(oe.isFitted());
        oe.fit(new Matrix(new double[][]{{1.0}}), null);
        assertTrue(oe.isFitted());
    }
}
