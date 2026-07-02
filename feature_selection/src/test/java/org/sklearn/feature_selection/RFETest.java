package org.sklearn.feature_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.linear_model.LinearRegression;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class RFETest {

    @Test
    void testReduceFeatures() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0},
            {2, 0, 0},
            {3, 1, 0},
            {4, 1, 0}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        LinearRegression lr = new LinearRegression();
        RFE rfe = new RFE(lr, 1);
        rfe.fit(X, y);
        Matrix Xr = rfe.transform(X);

        assertEquals(4, Xr.rows());
        assertEquals(1, Xr.cols());
        assertTrue(rfe.isFitted());
    }

    @Test
    void testSelectAllFeatures() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {3, 4}, {5, 6}
        });
        Vector y = new Vector(new double[]{1, 3, 5});

        LinearRegression lr = new LinearRegression();
        RFE rfe = new RFE(lr, 2);
        rfe.fit(X, y);
        Matrix Xr = rfe.transform(X);

        assertEquals(3, Xr.rows());
        assertEquals(2, Xr.cols());
    }

    @Test
    void testWithStep() {
        Matrix X = new Matrix(new double[][]{
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12}
        });
        Vector y = new Vector(new double[]{1, 5, 9});

        LinearRegression lr = new LinearRegression();
        RFE rfe = new RFE(lr, 1, 2);
        rfe.fit(X, y);
        int[] selected = rfe.getSelectedIndices();
        assertEquals(1, selected.length);
    }

    @Test
    void testRanking() {
        Matrix X = new Matrix(new double[][]{
            {1, 10, 100},
            {2, 20, 200},
            {3, 30, 300}
        });
        Vector y = new Vector(new double[]{1, 2, 3});

        LinearRegression lr = new LinearRegression();
        RFE rfe = new RFE(lr, 1);
        rfe.fit(X, y);
        int[] ranking = rfe.getRanking();
        assertEquals(3, ranking.length);
    }

    @Test
    void testGetSupport() {
        Matrix X = new Matrix(new double[][]{
            {0, 1}, {2, 3}, {4, 5}
        });
        Vector y = new Vector(new double[]{0, 2, 4});

        LinearRegression lr = new LinearRegression();
        RFE rfe = new RFE(lr, 1);
        rfe.fit(X, y);
        int[] support = rfe.getSupport();
        assertEquals(1, support.length);
    }

    @Test
    void testFitTransform() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0}, {2, 0, 0}, {3, 0, 1}, {4, 0, 1}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        LinearRegression lr = new LinearRegression();
        RFE rfe = new RFE(lr, 1);
        Matrix Xr = rfe.fitTransform(X, y);
        assertEquals(4, Xr.rows());
        assertEquals(1, Xr.cols());
    }

    @Test
    void testInverseTransformThrows() {
        Matrix X = new Matrix(new double[][]{
            {1, 0}, {2, 0}, {3, 1}, {4, 1}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        LinearRegression lr = new LinearRegression();
        RFE rfe = new RFE(lr, 1);
        rfe.fit(X, y);
        assertThrows(UnsupportedOperationException.class, () -> rfe.inverseTransform(X));
    }
}
