package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class BinarizerTest {

    @Test
    void testDefaultThreshold() {
        Binarizer b = new Binarizer();
        b.fit(new Matrix(new double[][]{{0.0}}), null);
        Matrix X = new Matrix(new double[][]{{-1.0}, {0.0}, {0.5}, {2.0}});
        Matrix Xb = b.transform(X);
        assertEquals(0.0, Xb.get(0, 0), 1e-10);
        assertEquals(0.0, Xb.get(1, 0), 1e-10);
        assertEquals(1.0, Xb.get(2, 0), 1e-10);
        assertEquals(1.0, Xb.get(3, 0), 1e-10);
    }

    @Test
    void testCustomThreshold() {
        Binarizer b = new Binarizer(1.5);
        b.fit(new Matrix(new double[][]{{0.0}}), null);
        Matrix X = new Matrix(new double[][]{{1.0}, {1.5}, {2.0}});
        Matrix Xb = b.transform(X);
        assertEquals(0.0, Xb.get(0, 0), 1e-10);
        assertEquals(0.0, Xb.get(1, 0), 1e-10);
        assertEquals(1.0, Xb.get(2, 0), 1e-10);
    }

    @Test
    void testTransformBeforeFitThrows() {
        Binarizer b = new Binarizer();
        assertThrows(IllegalStateException.class,
            () -> b.transform(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testInverseThrows() {
        Binarizer b = new Binarizer();
        b.fit(new Matrix(new double[][]{{0.0}}), null);
        assertThrows(UnsupportedOperationException.class,
            () -> b.inverseTransform(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testIsFitted() {
        Binarizer b = new Binarizer();
        assertFalse(b.isFitted());
        b.fit(new Matrix(new double[][]{{0.0}}), null);
        b.fit(new Matrix(new double[][]{{1.0}}), null);
        assertTrue(b.isFitted());
    }
}
