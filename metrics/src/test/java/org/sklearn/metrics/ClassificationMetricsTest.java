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
    void testMismatchedLengthsThrows() {
        assertThrows(IllegalArgumentException.class,
            () -> ClassificationMetrics.accuracyScore(
                new Vector(new double[]{0, 1}), new Vector(new double[]{0})));
    }
}
