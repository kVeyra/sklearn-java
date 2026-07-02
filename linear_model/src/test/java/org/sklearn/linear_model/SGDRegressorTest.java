package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class SGDRegressorTest {

    @Test
    void testSquaredError() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}, {5}});
        Vector y = new Vector(new double[]{2, 4, 6, 8, 10});
        SGDRegressor reg = new SGDRegressor("squared_error", "l2", 0.0001, 0.15, true, 500, 1e-3, "constant", 0.01, 0.5, 0.1, true, 42, false, false);
        reg.fit(X, y);
        Vector pred = reg.predict(new Matrix(new double[][]{{3}}));
        assertEquals(1, pred.size());
        assertTrue(reg.score(X, y) > 0.5);
    }

    @Test
    void testEpsilonInsensitive() {
        Matrix X = new Matrix(new double[][]{{1, 1}, {2, 2}, {3, 3}, {4, 4}});
        Vector y = new Vector(new double[]{1, 2, 3, 4});
        SGDRegressor reg = new SGDRegressor("epsilon_insensitive", "l2", 0.0001, 0.15, true, 500, 1e-3, "constant", 0.01, 0.5, 0.1, true, 42, false, false);
        reg.fit(X, y);
        assertTrue(reg.score(X, y) > 0);
    }

    @Test
    void testPredictBeforeFitThrows() {
        SGDRegressor reg = new SGDRegressor();
        assertThrows(IllegalStateException.class, () -> reg.predict(new Matrix(1, 1)));
    }
}
