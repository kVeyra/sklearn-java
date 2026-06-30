package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class NormalizerTest {

    @Test
    void testL2Norm() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0, 2.0},
            {4.0, 0.0, 3.0}
        });

        Normalizer normalizer = new Normalizer();
        normalizer.fit(X, null);
        Matrix Xn = normalizer.transform(X);

        for (int i = 0; i < Xn.rows(); i++) {
            double sumSq = 0.0;
            for (int j = 0; j < Xn.cols(); j++) {
                sumSq += Xn.get(i, j) * Xn.get(i, j);
            }
            assertEquals(1.0, sumSq, 1e-12);
        }
    }

    @Test
    void testL1Norm() {
        Matrix X = new Matrix(new double[][]{
            {1.0, -2.0, 3.0},
            {0.0, 5.0, 0.0}
        });

        Normalizer normalizer = new Normalizer(Normalizer.Norm.L1);
        normalizer.fit(X, null);
        Matrix Xn = normalizer.transform(X);

        for (int i = 0; i < Xn.rows(); i++) {
            double sumAbs = 0.0;
            for (int j = 0; j < Xn.cols(); j++) {
                sumAbs += Math.abs(Xn.get(i, j));
            }
            assertEquals(1.0, sumAbs, 1e-12);
        }
    }

    @Test
    void testMaxNorm() {
        Matrix X = new Matrix(new double[][]{
            {2.0, -4.0, 1.0},
            {-3.0, 0.0, 6.0}
        });

        Normalizer normalizer = new Normalizer(Normalizer.Norm.MAX);
        normalizer.fit(X, null);
        Matrix Xn = normalizer.transform(X);

        for (int i = 0; i < Xn.rows(); i++) {
            double maxAbs = 0.0;
            for (int j = 0; j < Xn.cols(); j++) {
                maxAbs = Math.max(maxAbs, Math.abs(Xn.get(i, j)));
            }
            assertEquals(1.0, maxAbs, 1e-12);
        }
    }

    @Test
    void testZeroRowIsUnchanged() {
        Matrix X = new Matrix(new double[][]{
            {0.0, 0.0, 0.0},
            {1.0, 2.0, 3.0}
        });

        Normalizer normalizer = new Normalizer();
        normalizer.fit(X, null);
        Matrix Xn = normalizer.transform(X);

        for (int j = 0; j < Xn.cols(); j++) {
            assertEquals(0.0, Xn.get(0, j), 1e-12);
        }

        double sumSq = 0.0;
        for (int j = 0; j < Xn.cols(); j++) {
            sumSq += Xn.get(1, j) * Xn.get(1, j);
        }
        assertEquals(1.0, sumSq, 1e-12);
    }

    @Test
    void testTransformReturnsNewCopy() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0}
        });

        Normalizer normalizer = new Normalizer();
        normalizer.fit(X, null);
        Matrix Xn = normalizer.transform(X);

        assertNotSame(X, Xn);

        Xn.set(0, 0, 999.0);
        assertEquals(1.0, X.get(0, 0), 1e-12);
    }

    @Test
    void testInverseTransformThrows() {
        Normalizer normalizer = new Normalizer();
        normalizer.fit(new Matrix(new double[][]{{1.0, 2.0}}), null);

        Matrix X = new Matrix(new double[][]{{1.0, 2.0}});
        assertThrows(UnsupportedOperationException.class,
            () -> normalizer.inverseTransform(X));
    }

    @Test
    void testDefaultConstructorUsesL2() {
        Normalizer normalizer = new Normalizer();
        assertEquals(Normalizer.Norm.L2, normalizer.getNorm());
    }

    @Test
    void testFitTransformConvenience() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0, 2.0},
            {4.0, 0.0, 3.0}
        });

        Normalizer normalizer = new Normalizer();
        Matrix Xn = normalizer.fitTransform(X, null);

        assertTrue(normalizer.isFitted());

        for (int i = 0; i < Xn.rows(); i++) {
            double sumSq = 0.0;
            for (int j = 0; j < Xn.cols(); j++) {
                sumSq += Xn.get(i, j) * Xn.get(i, j);
            }
            assertEquals(1.0, sumSq, 1e-12);
        }
    }

    @Test
    void testTransformFailsWhenNotFitted() {
        Normalizer normalizer = new Normalizer();
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}});
        assertThrows(IllegalStateException.class, () -> normalizer.transform(X));
    }

    @Test
    void testGetParameters() {
        Normalizer normalizer = new Normalizer(Normalizer.Norm.L1);
        normalizer.fit(new Matrix(new double[][]{{1.0, 2.0}}), null);

        Map<String, Object> params = normalizer.getParameters();
        assertEquals("L1", params.get("norm"));
        assertEquals(1, params.size());
    }
}
