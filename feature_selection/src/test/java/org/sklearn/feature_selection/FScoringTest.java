package org.sklearn.feature_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class FScoringTest {

    @Test
    void testFClassif() {
        Matrix X = new Matrix(new double[][]{
            {1, 0}, {2, 0}, {3, 1}, {4, 1}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        FScoring.FTestResult r = FScoring.fClassif(X, y);
        assertEquals(2, r.statistic.length);
        assertEquals(2, r.pvalue.length);
        assertTrue(r.statistic[0] > r.statistic[1]);
    }

    @Test
    void testFRegression() {
        Matrix X = new Matrix(new double[][]{
            {1, 10}, {2, 10}, {3, 10}, {4, 10}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        FScoring.FTestResult r = FScoring.fRegression(X, y);
        assertEquals(2, r.statistic.length);
        assertTrue(r.statistic[0] > r.statistic[1]);
    }

    @Test
    void testRRegression() {
        Matrix X = new Matrix(new double[][]{
            {1, 10}, {2, 10}, {3, 10}, {4, 10}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        double[] r = FScoring.rRegression(X, y);
        assertEquals(2, r.length);
        assertTrue(Math.abs(r[0]) > Math.abs(r[1]));
    }
}
