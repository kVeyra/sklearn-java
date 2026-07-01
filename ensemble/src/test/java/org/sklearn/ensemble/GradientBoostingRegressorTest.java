package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class GradientBoostingRegressorTest {

    @Test
    void testLSRegression() {
        Matrix X = new Matrix(new double[][]{
            {1}, {2}, {3}, {4}, {5}, {6}
        });
        Vector y = new Vector(new double[]{2, 4, 6, 8, 10, 12});

        GradientBoostingRegressor gbr = new GradientBoostingRegressor("ls", 0.1, 50, 3, 2, 1, 1.0, 0.9, 42);
        gbr.fit(X, y);
        Vector pred = gbr.predict(new Matrix(new double[][]{{3}, {7}}));
        assertNotNull(pred);
        assertEquals(2, pred.size());
    }

    @Test
    void testLADRegression() {
        Matrix X = new Matrix(new double[][]{
            {1, 1}, {2, 2}, {3, 3}, {4, 4}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        GradientBoostingRegressor gbr = new GradientBoostingRegressor("lad", 0.1, 30, 2, 2, 1, 1.0, 0.9, 42);
        gbr.fit(X, y);
        double score = gbr.score(X, y);
        assertTrue(score > 0.8);
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1}, {2}, {3}, {4}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        GradientBoostingRegressor gbr = new GradientBoostingRegressor(20, 3, 42);
        gbr.fit(X, y);
        assertTrue(gbr.score(X, y) > 0.5);
    }
}
