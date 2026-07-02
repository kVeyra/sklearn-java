package org.sklearn.cluster;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class MeanShiftTest {

    @Test
    void testBasicClustering() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {1.1, 1.1}, {5.0, 5.0}, {5.1, 5.1}
        });
        MeanShift ms = new MeanShift(1.0, 300);
        ms.fit(X);
        assertTrue(ms.isFitted());
        int[] labels = ms.getLabels();
        assertEquals(4, labels.length);
        assertEquals(labels[0], labels[1]);
        assertEquals(labels[2], labels[3]);
    }

    @Test
    void testGetClusterCenters() {
        Matrix X = new Matrix(new double[][]{{0.0, 0.0}, {0.1, 0.1}, {5.0, 5.0}});
        MeanShift ms = new MeanShift(1.0, 300);
        ms.fit(X);
        Matrix centers = ms.getClusterCenters();
        assertNotNull(centers);
        assertTrue(centers.rows() >= 2);
        assertEquals(2, centers.cols());
    }

    @Test
    void testFitPredict() {
        Matrix X = new Matrix(new double[][]{
            {0.0, 0.0}, {0.1, 0.1}, {5.0, 5.0}, {5.1, 5.1}
        });
        MeanShift ms = new MeanShift(1.0, 300);
        int[] labels = ms.fitPredict(X);
        assertEquals(4, labels.length);
    }

    @Test
    void testAutoBandwidth() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {10.0, 10.0}, {11.0, 11.0}
        });
        MeanShift ms = new MeanShift(-1.0, 300);
        ms.fit(X);
        assertTrue(ms.isFitted());
        assertNotNull(ms.getLabels());
    }

    @Test
    void testGetParameters() {
        MeanShift ms = new MeanShift(2.5, 100);
        Map<String, Object> params = ms.getParameters();
        assertEquals(2.5, params.get("bandwidth"));
        assertEquals(100, params.get("max_iter"));
    }
}
