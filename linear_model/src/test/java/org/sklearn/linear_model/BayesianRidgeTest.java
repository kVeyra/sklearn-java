package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class BayesianRidgeTest {

    @Test
    void testFitAndPredict() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}, {5}});
        Vector y = new Vector(new double[]{2, 4, 6, 8, 10});
        BayesianRidge br = new BayesianRidge();
        br.fit(X, y);
        Vector pred = br.predict(new Matrix(new double[][]{{6}}));
        assertEquals(1, pred.size());
        assertFalse(Double.isNaN(pred.get(0)));
        assertTrue(br.getLambda() > 0);
        assertFalse(Double.isNaN(pred.get(0)));
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{{1, 1}, {2, 2}, {3, 3}, {4, 4}});
        Vector y = new Vector(new double[]{1, 2, 3, 4});
        BayesianRidge br = new BayesianRidge();
        br.fit(X, y);
        double s = br.score(X, y);
        assertFalse(Double.isNaN(s));
    }

    @Test
    void testPredictBeforeFitThrows() {
        BayesianRidge br = new BayesianRidge();
        assertThrows(IllegalStateException.class, () -> br.predict(new Matrix(1, 1)));
    }
}
