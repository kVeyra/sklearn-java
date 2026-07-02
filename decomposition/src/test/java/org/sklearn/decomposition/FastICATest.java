package org.sklearn.decomposition;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class FastICATest {

    @Test
    void testFitTransform() {
        Matrix X = new Matrix(new double[][]{
            {1, 2},
            {3, 4},
            {5, 6},
            {7, 8}
        });
        FastICA ica = new FastICA(2);
        ica.fit(X, null);
        Matrix sources = ica.transform(X);
        assertEquals(4, sources.rows());
        assertEquals(2, sources.cols());
    }

    @Test
    void testMixingMatrix() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1},
            {1, 1, 0}
        });
        FastICA ica = new FastICA(2);
        ica.fit(X, null);
        assertNotNull(ica.getMixing());
        assertNotNull(ica.getUnmixing());
    }

    @Test
    void testInverseTransform() {
        Matrix X = new Matrix(new double[][]{
            {1, 2},
            {3, 4},
            {5, 6},
            {7, 8}
        });
        FastICA ica = new FastICA(2);
        ica.fit(X, null);
        Matrix sources = ica.transform(X);
        Matrix reconstructed = ica.inverseTransform(sources);
        assertEquals(X.rows(), reconstructed.rows());
        assertEquals(X.cols(), reconstructed.cols());
    }
}
