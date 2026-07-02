package org.sklearn.metrics;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class RankingMetricsTest {

    @Test
    void testRocCurve() {
        Vector yTrue = new Vector(new double[]{1, 1, 0, 0});
        Vector yScore = new Vector(new double[]{0.9, 0.8, 0.3, 0.2});
        RankingMetrics.RocCurve curve = RankingMetrics.rocCurve(yTrue, yScore);
        assertNotNull(curve);
        assertTrue(curve.fpr.length > 0);
        assertTrue(curve.tpr.length > 0);
    }

    @Test
    void testAuc() {
        double[] x = {0, 0.5, 1};
        double[] y = {0, 0.5, 1};
        assertEquals(0.5, RankingMetrics.auc(x, y), 1e-10);
    }

    @Test
    void testRocAucScore() {
        Vector yTrue = new Vector(new double[]{1, 1, 0, 0});
        Vector yScore = new Vector(new double[]{0.9, 0.8, 0.3, 0.2});
        double auc = RankingMetrics.rocAucScore(yTrue, yScore);
        assertTrue(auc > 0.9);
    }

    @Test
    void testRocAucScorePerfect() {
        Vector yTrue = new Vector(new double[]{1, 1, 0, 0});
        Vector yScore = new Vector(new double[]{1.0, 0.9, 0.1, 0.0});
        assertEquals(1.0, RankingMetrics.rocAucScore(yTrue, yScore), 1e-10);
    }

    @Test
    void testRocAucScoreWorst() {
        Vector yTrue = new Vector(new double[]{1, 1, 0, 0});
        Vector yScore = new Vector(new double[]{0.0, 0.1, 0.9, 1.0});
        assertEquals(0.0, RankingMetrics.rocAucScore(yTrue, yScore), 1e-10);
    }

    @Test
    void testPrecisionRecallCurve() {
        Vector yTrue = new Vector(new double[]{1, 1, 0, 0});
        Vector yScore = new Vector(new double[]{0.9, 0.8, 0.3, 0.2});
        RankingMetrics.PrCurve pr = RankingMetrics.precisionRecallCurve(yTrue, yScore);
        assertNotNull(pr);
        assertTrue(pr.precision.length > 0);
    }

    @Test
    void testAveragePrecisionScore() {
        Vector yTrue = new Vector(new double[]{1, 1, 0, 0});
        Vector yScore = new Vector(new double[]{0.9, 0.8, 0.3, 0.2});
        double ap = RankingMetrics.averagePrecisionScore(yTrue, yScore);
        assertTrue(ap > 0);
    }

    @Test
    void testTopKPrecision() {
        Vector yTrue = new Vector(new double[]{1, 1, 0, 0});
        Vector yScore = new Vector(new double[]{0.9, 0.2, 0.8, 0.1});
        double p = RankingMetrics.topKPrecision(yTrue, yScore, 2);
        assertEquals(0.5, p, 1e-10);
    }

    @Test
    void testTopKRecall() {
        Vector yTrue = new Vector(new double[]{1, 1, 0, 0});
        Vector yScore = new Vector(new double[]{0.9, 0.2, 0.8, 0.1});
        double r = RankingMetrics.topKRecall(yTrue, yScore, 2);
        assertEquals(0.5, r, 1e-10);
    }

    @Test
    void testDCG() {
        Vector yTrue = new Vector(new double[]{3, 2, 3, 0, 1, 2});
        Vector yScore = new Vector(new double[]{0.9, 0.8, 0.7, 0.6, 0.5, 0.4});
        double dcg = RankingMetrics.dcgScore(yTrue, yScore);
        assertTrue(dcg > 0);
    }

    @Test
    void testNDCG() {
        Vector yTrue = new Vector(new double[]{3, 2, 3, 0, 1, 2});
        Vector yScore = new Vector(new double[]{0.9, 0.8, 0.7, 0.6, 0.5, 0.4});
        double ndcg = RankingMetrics.ndcgScore(yTrue, yScore);
        assertTrue(ndcg > 0 && ndcg <= 1.0);
    }

    @Test
    void testNDCGPerfect() {
        Vector yTrue = new Vector(new double[]{3, 2, 3});
        Vector yScore = new Vector(new double[]{1.0, 0.5, 0.9});
        assertEquals(1.0, RankingMetrics.ndcgScore(yTrue, yScore), 1e-10);
    }

    @Test
    void testLengthMismatchThrows() {
        assertThrows(IllegalArgumentException.class, () ->
            RankingMetrics.rocAucScore(new Vector(2), new Vector(3)));
    }
}
