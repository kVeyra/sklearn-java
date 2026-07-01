package org.sklearn.metrics;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class RegressionMetricsTest {

    @Test
    void testMeanAbsoluteError() {
        Vector yTrue = new Vector(new double[]{1.0, 2.0, 3.0});
        Vector yPred = new Vector(new double[]{1.5, 2.5, 2.5});
        assertEquals(0.5, RegressionMetrics.meanAbsoluteError(yTrue, yPred), 1e-10);
    }

    @Test
    void testMeanSquaredError() {
        Vector yTrue = new Vector(new double[]{1.0, 2.0, 3.0});
        Vector yPred = new Vector(new double[]{1.0, 3.0, 3.0});
        assertEquals(1.0 / 3, RegressionMetrics.meanSquaredError(yTrue, yPred), 1e-10);
    }

    @Test
    void testRootMeanSquaredError() {
        Vector yTrue = new Vector(new double[]{0.0, 1.0});
        Vector yPred = new Vector(new double[]{1.0, 1.0});
        assertEquals(Math.sqrt(0.5), RegressionMetrics.rootMeanSquaredError(yTrue, yPred), 1e-10);
    }

    @Test
    void testR2Score() {
        Vector yTrue = new Vector(new double[]{1.0, 2.0, 3.0});
        Vector yPred = new Vector(new double[]{1.0, 2.0, 3.0});
        assertEquals(1.0, RegressionMetrics.r2Score(yTrue, yPred), 1e-10);
    }

    @Test
    void testR2ScoreNegative() {
        Vector yTrue = new Vector(new double[]{1.0, 2.0, 3.0});
        Vector yPred = new Vector(new double[]{3.0, 2.0, 1.0});
        assertTrue(RegressionMetrics.r2Score(yTrue, yPred) < 0);
    }

    @Test
    void testMaxError() {
        Vector yTrue = new Vector(new double[]{1.0, 5.0, 3.0});
        Vector yPred = new Vector(new double[]{1.0, 2.0, 3.0});
        assertEquals(3.0, RegressionMetrics.maxError(yTrue, yPred), 1e-10);
    }

    @Test
    void testMeanAbsolutePercentageError() {
        Vector yTrue = new Vector(new double[]{10.0, 20.0, 30.0});
        Vector yPred = new Vector(new double[]{11.0, 20.0, 30.0});
        double mape = RegressionMetrics.meanAbsolutePercentageError(yTrue, yPred);
        assertTrue(mape > 0);
    }

    @Test
    void testExplainedVarianceScore() {
        Vector yTrue = new Vector(new double[]{1.0, 2.0, 3.0});
        Vector yPred = new Vector(new double[]{1.0, 2.0, 3.0});
        assertEquals(1.0, RegressionMetrics.explainedVarianceScore(yTrue, yPred), 1e-10);
    }

    @Test
    void testMedianAbsoluteError() {
        Vector yTrue = new Vector(new double[]{1.0, 2.0, 10.0});
        Vector yPred = new Vector(new double[]{2.0, 3.0, 11.0});
        assertEquals(1.0, RegressionMetrics.medianAbsoluteError(yTrue, yPred), 1e-10);
    }

    @Test
    void testLengthMismatchThrows() {
        assertThrows(IllegalArgumentException.class, () ->
            RegressionMetrics.meanAbsoluteError(new Vector(3), new Vector(4)));
    }
}
