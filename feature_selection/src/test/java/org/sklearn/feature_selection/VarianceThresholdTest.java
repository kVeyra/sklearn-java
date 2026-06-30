package org.sklearn.feature_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class VarianceThresholdTest {

    @Test
    void testRemoveZeroVariance() {
        Matrix X = new Matrix(new double[][]{
            {1, 1, 1}, {2, 1, 2}, {3, 1, 3}
        });

        VarianceThreshold vt = new VarianceThreshold(0.0);
        Matrix Xr = vt.fitTransform(X);

        // Column 1 (index 1) has zero variance, should be removed
        assertEquals(2, Xr.cols());
    }

    @Test
    void testKeepAll() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {3, 4}, {5, 6}
        });

        VarianceThreshold vt = new VarianceThreshold(0.0);
        Matrix Xr = vt.fitTransform(X);

        assertEquals(2, Xr.cols());
        assertEquals(3, Xr.rows());
    }

    @Test
    void testHighThreshold() {
        Matrix X = new Matrix(new double[][]{
            {1, 1}, {2, 2}, {3, 3}
        });

        VarianceThreshold vt = new VarianceThreshold(0.7);
        Matrix Xr = vt.fitTransform(X);

        // Column 0 has var ~0.67, column 1 has var ~0.67
        // Both below 0.7
        assertEquals(0, Xr.cols());
    }

    @Test
    void testTransformBeforeFitThrows() {
        VarianceThreshold vt = new VarianceThreshold(0.1);
        assertThrows(IllegalStateException.class,
            () -> vt.transform(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testGetParameters() {
        VarianceThreshold vt = new VarianceThreshold(0.5);
        assertEquals(0.5, vt.getParameters().get("threshold"));
    }
}
