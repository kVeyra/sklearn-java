package org.sklearn.metrics;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class ClassificationMetricsTest {

    @Test
    void testAccuracyPerfect() {
        Vector yTrue = new Vector(new double[]{0, 1, 0, 1});
        Vector yPred = new Vector(new double[]{0, 1, 0, 1});
        assertEquals(1.0, ClassificationMetrics.accuracyScore(yTrue, yPred));
    }

    @Test
    void testAccuracyHalf() {
        Vector yTrue = new Vector(new double[]{0, 0, 1, 1});
        Vector yPred = new Vector(new double[]{0, 1, 0, 1});
        assertEquals(0.5, ClassificationMetrics.accuracyScore(yTrue, yPred));
    }

    @Test
    void testConfusionMatrix() {
        Vector yTrue = new Vector(new double[]{0, 0, 1, 1});
        Vector yPred = new Vector(new double[]{0, 1, 0, 1});
        Matrix cm = ClassificationMetrics.confusionMatrix(yTrue, yPred, 2);
        assertEquals(1.0, cm.get(0, 0));
        assertEquals(1.0, cm.get(0, 1));
        assertEquals(1.0, cm.get(1, 0));
        assertEquals(1.0, cm.get(1, 1));
    }

    @Test
    void testPrecisionRecall() {
        Vector yTrue = new Vector(new double[]{1, 1, 0, 0, 1});
        Vector yPred = new Vector(new double[]{1, 0, 0, 0, 1});
        assertEquals(1.0, ClassificationMetrics.precisionScore(yTrue, yPred, 1));
        assertEquals(2.0 / 3, ClassificationMetrics.recallScore(yTrue, yPred, 1), 1e-10);
        assertEquals(0.8, ClassificationMetrics.f1Score(yTrue, yPred, 1), 1e-10);
    }

    @Test
    void testFbetaScore() {
        Vector yTrue = new Vector(new double[]{1, 1, 0, 0, 1});
        Vector yPred = new Vector(new double[]{1, 0, 0, 0, 1});
        double fb = ClassificationMetrics.fbetaScore(yTrue, yPred, 1, 2.0);
        assertTrue(fb > 0);
        double p = ClassificationMetrics.precisionScore(yTrue, yPred, 1);
        double r = ClassificationMetrics.recallScore(yTrue, yPred, 1);
        double expected = (5 * p * r) / (4 * p + r);
        assertEquals(expected, fb, 1e-10);
    }

    @Test
    void testZeroOneLoss() {
        Vector yTrue = new Vector(new double[]{0, 0, 1, 1});
        Vector yPred = new Vector(new double[]{0, 1, 0, 1});
        assertEquals(0.5, ClassificationMetrics.zeroOneLoss(yTrue, yPred));
    }

    @Test
    void testHammingLoss() {
        Vector yTrue = new Vector(new double[]{0, 0, 1, 1});
        Vector yPred = new Vector(new double[]{0, 1, 1, 0});
        assertEquals(0.5, ClassificationMetrics.hammingLoss(yTrue, yPred));
    }

    @Test
    void testJaccardScore() {
        Vector yTrue = new Vector(new double[]{1, 1, 0, 0, 1});
        Vector yPred = new Vector(new double[]{1, 0, 0, 0, 1});
        assertEquals(2.0 / 3, ClassificationMetrics.jaccardScore(yTrue, yPred, 1), 1e-10);
    }

    @Test
    void testMatthewsCorrcoef() {
        Vector yTrue = new Vector(new double[]{1, 1, 0, 0, 1, 0});
        Vector yPred = new Vector(new double[]{1, 0, 0, 0, 1, 1});
        double mcc = ClassificationMetrics.matthewsCorrcoef(yTrue, yPred);
        assertTrue(mcc > -1.01 && mcc < 1.01);
    }

    @Test
    void testCohenKappaScore() {
        Vector yTrue = new Vector(new double[]{0, 0, 1, 1});
        Vector yPred = new Vector(new double[]{0, 1, 0, 1});
        double kappa = ClassificationMetrics.cohenKappaScore(yTrue, yPred, 2);
        assertEquals(0.0, kappa, 1e-10);
    }

    @Test
    void testLogLoss() {
        Vector yTrue = new Vector(new double[]{1, 0, 1});
        Vector yPred = new Vector(new double[]{0.9, 0.1, 0.8});
        double loss = ClassificationMetrics.logLoss(yTrue, yPred);
        assertTrue(loss > 0);
    }

    @Test
    void testBalancedAccuracyScore() {
        Vector yTrue = new Vector(new double[]{1, 1, 0, 0, 0});
        Vector yPred = new Vector(new double[]{1, 0, 0, 0, 0});
        double ba = ClassificationMetrics.balancedAccuracyScore(yTrue, yPred);
        assertTrue(ba > 0 && ba < 1);
    }

    @Test
    void testTopKAccuracyScore() {
        Vector yTrue = new Vector(new double[]{0, 1});
        Matrix yScore = new Matrix(new double[][]{{0.7, 0.3}, {0.4, 0.6}});
        assertEquals(1.0, ClassificationMetrics.topKAccuracyScore(yTrue, yScore, 2));
    }

    @Test
    void testClassificationReport() {
        Vector yTrue = new Vector(new double[]{0, 0, 1, 1});
        Vector yPred = new Vector(new double[]{0, 1, 0, 1});
        String report = ClassificationMetrics.classificationReport(yTrue, yPred);
        assertTrue(report.contains("class 0"));
        assertTrue(report.contains("class 1"));
    }

    @Test
    void testMismatchedLengthsThrows() {
        assertThrows(IllegalArgumentException.class,
            () -> ClassificationMetrics.accuracyScore(
                new Vector(new double[]{0, 1}), new Vector(new double[]{0})));
    }
}
