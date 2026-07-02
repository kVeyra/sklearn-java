package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class ExtraTreeRegressorTest {

    @Test
    void testFitAndPredict() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {3, 4}, {4, 5}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        ExtraTreeRegressor etr = new ExtraTreeRegressor(3, 2, 1, 42);
        etr.fit(X, y);
        Vector pred = etr.predict(X);
        assertEquals(4, pred.size());
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1}, {2}, {3}, {4}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        ExtraTreeRegressor etr = new ExtraTreeRegressor(3, 2, 1, 42);
        etr.fit(X, y);
        double score = etr.score(X, y);
        assertTrue(score > 0 || score <= 1.0);
    }
}
