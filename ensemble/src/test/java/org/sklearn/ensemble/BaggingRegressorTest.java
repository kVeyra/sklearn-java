package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeRegressor;

import static org.junit.jupiter.api.Assertions.*;

class BaggingRegressorTest {

    @Test
    void testRegression() {
        Matrix X = new Matrix(new double[][]{
            {1}, {2}, {3}, {4}, {5}, {6}
        });
        Vector y = new Vector(new double[]{2, 4, 6, 8, 10, 12});

        BaggingRegressor br = new BaggingRegressor(
            new DecisionTreeRegressor(3, 2, 1), 10, 42);
        br.fit(X, y);
        Vector pred = br.predict(X);
        assertEquals(6, pred.size());
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1}, {2}, {3}, {4}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        BaggingRegressor br = new BaggingRegressor(
            new DecisionTreeRegressor(3, 2, 1), 10, 42);
        br.fit(X, y);
        assertTrue(br.score(X, y) > 0);
    }
}
