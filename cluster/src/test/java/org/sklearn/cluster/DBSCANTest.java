package org.sklearn.cluster;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class DBSCANTest {

    @Test
    void testTwoClustersNoNoise() {
        Matrix X = new Matrix(new double[][]{
            {0, 0}, {0.1, 0.1}, {-0.1, -0.1},
            {5, 5}, {5.1, 5.1}, {4.9, 4.9}
        });

        DBSCAN dbscan = new DBSCAN(0.5, 2);
        dbscan.fit(X);

        int[] labels = dbscan.getLabels();
        assertEquals(6, labels.length);

        // Points within each cluster should have the same label
        assertEquals(labels[0], labels[1]);
        assertEquals(labels[0], labels[2]);
        assertEquals(labels[3], labels[4]);
        assertEquals(labels[3], labels[5]);
    }

    @Test
    void testWithNoise() {
        Matrix X = new Matrix(new double[][]{
            {0, 0}, {0.1, 0.1},
            {10, 10}, // far away = noise
            {5, 5}, {5.1, 5.1}
        });

        DBSCAN dbscan = new DBSCAN(0.3, 2);
        dbscan.fit(X);

        int[] labels = dbscan.getLabels();
        assertEquals(labels[0], labels[1]); // close pair should cluster
        assertEquals(0, labels[2]); // isolated point = noise
    }

    @Test
    void testSingleCluster() {
        Matrix X = new Matrix(new double[][]{
            {1, 1}, {1.1, 1.1}, {1.2, 1.2}, {0.9, 0.9}
        });

        DBSCAN dbscan = new DBSCAN(0.5, 2);
        int[] labels = dbscan.fitPredict(X);

        for (int i = 0; i < labels.length; i++) {
            assertTrue(labels[i] > 0);
        }
    }

    @Test
    void testAllNoise() {
        Matrix X = new Matrix(new double[][]{
            {0, 0}, {10, 10}, {20, 20}
        });

        DBSCAN dbscan = new DBSCAN(0.1, 2);
        dbscan.fit(X);

        int[] labels = dbscan.getLabels();
        for (int label : labels) {
            assertEquals(0, label);
        }
    }

    @Test
    void testGetParameters() {
        DBSCAN dbscan = new DBSCAN(0.5, 10);
        var params = dbscan.getParameters();
        assertEquals(0.5, params.get("eps"));
        assertEquals(10, params.get("min_samples"));
    }
}
