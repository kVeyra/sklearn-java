package org.sklearn.feature_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.linear_model.LinearRegression;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class SelectFromModelTest {

    @Test
    void testSelectByMeanThreshold() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0},
            {2, 0, 0},
            {3, 0, 1},
            {4, 0, 1}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        LinearRegression lr = new LinearRegression();
        SelectFromModel sfm = new SelectFromModel(lr);
        sfm.fit(X, y);
        Matrix Xr = sfm.transform(X);

        assertEquals(4, Xr.rows());
        assertTrue(Xr.cols() >= 1 && Xr.cols() <= 3);
        assertTrue(sfm.isFitted());
    }

    @Test
    void testSelectByCustomThreshold() {
        Matrix X = new Matrix(new double[][]{
            {1, 10},
            {2, 20},
            {3, 30}
        });
        Vector y = new Vector(new double[]{1, 2, 3});

        LinearRegression lr = new LinearRegression();
        SelectFromModel sfm = new SelectFromModel(lr, 0.5, "mean");
        sfm.fit(X, y);
        Matrix Xr = sfm.transform(X);

        assertEquals(3, Xr.rows());
    }

    @Test
    void testSelectByMedianThreshold() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0},
            {2, 0, 0},
            {3, 1, 1},
            {4, 1, 1}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        LinearRegression lr = new LinearRegression();
        SelectFromModel sfm = new SelectFromModel(lr, null, "median");
        sfm.fit(X, y);
        Matrix Xr = sfm.transform(X);

        assertEquals(4, Xr.rows());
        assertTrue(Xr.cols() >= 1);
    }

    @Test
    void testGetSupport() {
        Matrix X = new Matrix(new double[][]{
            {1, 0}, {2, 0}, {3, 1}, {4, 1}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        LinearRegression lr = new LinearRegression();
        SelectFromModel sfm = new SelectFromModel(lr);
        sfm.fit(X, y);
        int[] support = sfm.getSupport();
        assertTrue(support.length >= 1);
    }

    @Test
    void testFitTransform() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0}, {2, 0, 0}, {3, 1, 0}, {4, 1, 0}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        LinearRegression lr = new LinearRegression();
        SelectFromModel sfm = new SelectFromModel(lr);
        Matrix Xr = sfm.fitTransform(X, y);
        assertEquals(4, Xr.rows());
        assertTrue(Xr.cols() >= 1);
    }

    @Test
    void testInverseTransformThrows() {
        Matrix X = new Matrix(new double[][]{
            {1, 0}, {2, 0}, {3, 1}, {4, 1}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        LinearRegression lr = new LinearRegression();
        SelectFromModel sfm = new SelectFromModel(lr);
        sfm.fit(X, y);
        assertThrows(UnsupportedOperationException.class, () -> sfm.inverseTransform(X));
    }
}
