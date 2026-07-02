package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class PassiveAggressiveRegressorTest {

    @Test
    void testRegression() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}});
        Vector y = new Vector(new double[]{1, 2, 3, 4});
        PassiveAggressiveRegressor pa = new PassiveAggressiveRegressor(1.0, true, 500, 1e-3, "epsilon_insensitive", 0.1, true, 42);
        pa.fit(X, y);
        Vector pred = pa.predict(X);
        assertEquals(4, pred.size());
    }

    @Test
    void testSquaredEpsilon() {
        Matrix X = new Matrix(new double[][]{{1, 1}, {2, 2}, {3, 3}, {4, 4}});
        Vector y = new Vector(new double[]{1, 2, 3, 4});
        PassiveAggressiveRegressor pa = new PassiveAggressiveRegressor(1.0, true, 500, 1e-3, "squared_epsilon_insensitive", 0.1, true, 42);
        pa.fit(X, y);
        assertFalse(Double.isNaN(pa.score(X, y)));
    }

    @Test
    void testPredictBeforeFitThrows() {
        PassiveAggressiveRegressor pa = new PassiveAggressiveRegressor();
        assertThrows(IllegalStateException.class, () -> pa.predict(new Matrix(1, 1)));
    }
}
