package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class PolynomialFeaturesTest {

    @Test
    void testDegree2WithBias() {
        Matrix X = new Matrix(new double[][]{{2.0, 3.0}});
        PolynomialFeatures pf = new PolynomialFeatures(2, true);
        pf.fit(X, null);
        Matrix Xp = pf.transform(X);
        assertEquals(1, Xp.rows());
        assertEquals(6, Xp.cols());
        assertEquals(1.0, Xp.get(0, 0), 1e-10);
        assertEquals(2.0, Xp.get(0, 1), 1e-10);
        assertEquals(3.0, Xp.get(0, 2), 1e-10);
        assertEquals(4.0, Xp.get(0, 3), 1e-10);
        assertEquals(6.0, Xp.get(0, 4), 1e-10);
        assertEquals(9.0, Xp.get(0, 5), 1e-10);
    }

    @Test
    void testDegree2WithoutBias() {
        Matrix X = new Matrix(new double[][]{{2.0, 3.0}});
        PolynomialFeatures pf = new PolynomialFeatures(2, false);
        pf.fit(X, null);
        Matrix Xp = pf.transform(X);
        assertEquals(5, Xp.cols());
    }

    @Test
    void testInteractionOnly() {
        Matrix X = new Matrix(new double[][]{{2.0, 3.0, 4.0}});
        PolynomialFeatures pf = new PolynomialFeatures(2, false, true);
        pf.fit(X, null);
        Matrix Xp = pf.transform(X);
        assertEquals(1, Xp.rows());
        assertEquals(6, Xp.cols());
        assertEquals(2.0, Xp.get(0, 0), 1e-10); // [0]
        assertEquals(3.0, Xp.get(0, 1), 1e-10); // [1]
        assertEquals(4.0, Xp.get(0, 2), 1e-10); // [2]
        assertEquals(6.0, Xp.get(0, 3), 1e-10); // [0,1] = 2*3
        assertEquals(8.0, Xp.get(0, 4), 1e-10); // [0,2] = 2*4
        assertEquals(12.0, Xp.get(0, 5), 1e-10); // [1,2] = 3*4
    }

    @Test
    void testTransformBeforeFitThrows() {
        PolynomialFeatures pf = new PolynomialFeatures(2, true);
        assertThrows(IllegalStateException.class,
            () -> pf.transform(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testInverseThrows() {
        PolynomialFeatures pf = new PolynomialFeatures(2, true);
        pf.fit(new Matrix(new double[][]{{1.0, 2.0}}), null);
        assertThrows(UnsupportedOperationException.class,
            () -> pf.inverseTransform(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testIsFitted() {
        PolynomialFeatures pf = new PolynomialFeatures(2, true);
        assertFalse(pf.isFitted());
        pf.fit(new Matrix(new double[][]{{1.0}}), null);
        assertTrue(pf.isFitted());
    }
}
