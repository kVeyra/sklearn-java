package org.sklearn.decomposition;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class TruncatedSVDTest {

    @Test
    void testFitTransform() {
        Matrix X = new Matrix(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
        TruncatedSVD svd = new TruncatedSVD(2);
        svd.fit(X, null);
        Matrix Xr = svd.transform(X);
        assertEquals(4, Xr.rows());
        assertEquals(2, Xr.cols());
    }

    @Test
    void testInverseTransform() {
        Matrix X = new Matrix(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });
        TruncatedSVD svd = new TruncatedSVD(2);
        svd.fit(X, null);
        Matrix Xr = svd.transform(X);
        Matrix Xinv = svd.inverseTransform(Xr);
        assertEquals(3, Xinv.rows());
        assertEquals(3, Xinv.cols());
    }

    @Test
    void testComponents() {
        Matrix X = new Matrix(new double[][]{{1, 0}, {0, 1}, {1, 1}});
        TruncatedSVD svd = new TruncatedSVD(1);
        svd.fit(X, null);
        assertNotNull(svd.getComponents());
        assertNotNull(svd.getSingularValues());
    }
}
