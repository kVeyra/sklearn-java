package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class ARDRegressionTest {

    @Test
    void testFitPredict() {
        Matrix X = new Matrix(new double[][]{
            {1, 2},
            {2, 3},
            {3, 4},
            {4, 5},
            {5, 6}
        });
        Vector y = new Vector(new double[]{3, 5, 7, 9, 11});

        ARDRegression ard = new ARDRegression();
        ard.fit(X, y);
        Vector preds = ard.predict(X);

        assertEquals(5, preds.size());
        assertNotNull(ard.getCoef());
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{{1, 0}, {2, 0}, {3, 0}, {4, 0}});
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        ARDRegression ard = new ARDRegression();
        ard.fit(X, y);
        double score = ard.score(X, y);
        assertFalse(Double.isNaN(score));
    }
}
