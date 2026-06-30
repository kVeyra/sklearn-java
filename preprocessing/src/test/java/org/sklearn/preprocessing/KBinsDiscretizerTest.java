package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class KBinsDiscretizerTest {

    @Test
    void testUniformOrdinal() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}, {4.0}});
        KBinsDiscretizer kbd = new KBinsDiscretizer(4, "uniform", true);
        kbd.fit(X, null);
        Matrix Xt = kbd.transform(X);
        assertEquals(4, Xt.rows());
        assertEquals(1, Xt.cols());
        assertTrue(Xt.get(0, 0) >= 0);
        assertTrue(Xt.get(3, 0) <= 3);
    }

    @Test
    void testQuantileOrdinal() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}});
        KBinsDiscretizer kbd = new KBinsDiscretizer(3, "quantile", true);
        kbd.fit(X, null);
        Matrix Xt = kbd.transform(X);
        assertEquals(5, Xt.rows());
        assertEquals(1, Xt.cols());
    }

    @Test
    void testUniformOneHot() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}});
        KBinsDiscretizer kbd = new KBinsDiscretizer(2, "uniform", false);
        kbd.fit(X, null);
        Matrix Xt = kbd.transform(X);
        assertEquals(2, Xt.cols());
    }

    @Test
    void testTransformBeforeFitThrows() {
        KBinsDiscretizer kbd = new KBinsDiscretizer(3, "uniform", true);
        assertThrows(IllegalStateException.class,
            () -> kbd.transform(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testInverseThrows() {
        KBinsDiscretizer kbd = new KBinsDiscretizer(3, "uniform", true);
        kbd.fit(new Matrix(new double[][]{{1.0}, {2.0}, {3.0}}), null);
        assertThrows(UnsupportedOperationException.class,
            () -> kbd.inverseTransform(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testIsFitted() {
        KBinsDiscretizer kbd = new KBinsDiscretizer(3, "uniform", true);
        assertFalse(kbd.isFitted());
        kbd.fit(new Matrix(new double[][]{{1.0}, {2.0}}), null);
        assertTrue(kbd.isFitted());
    }

    @Test
    void testInvalidStrategy() {
        assertThrows(IllegalArgumentException.class,
            () -> new KBinsDiscretizer(3, "invalid", true));
    }

    @Test
    void testInvalidNBins() {
        assertThrows(IllegalArgumentException.class,
            () -> new KBinsDiscretizer(1, "uniform", true));
    }
}
