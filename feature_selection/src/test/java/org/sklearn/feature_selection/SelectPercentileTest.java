package org.sklearn.feature_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class SelectPercentileTest {

    @Test
    void testSelectFeatures() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0}, {2, 0, 0}, {3, 1, 1}, {4, 1, 1}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        SelectPercentile selector = new SelectPercentile(50, "f_classif");
        selector.fit(X, y);
        Matrix Xr = selector.transform(X);
        assertEquals(4, Xr.rows());
        assertTrue(Xr.cols() >= 1);
    }

    @Test
    void testGetSupport() {
        Matrix X = new Matrix(new double[][]{
            {1, 0}, {2, 0}, {3, 1}, {4, 1}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        SelectPercentile selector = new SelectPercentile(100, "f_classif");
        selector.fit(X, y);
        int[] support = selector.getSupport();
        assertEquals(2, support.length);
    }

    @Test
    void testFitTransform() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0}, {2, 0, 0}, {3, 1, 0}, {4, 1, 0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        SelectPercentile selector = new SelectPercentile(50, "f_classif");
        Matrix Xr = selector.fitTransform(X, y);
        assertEquals(4, Xr.rows());
    }
}
