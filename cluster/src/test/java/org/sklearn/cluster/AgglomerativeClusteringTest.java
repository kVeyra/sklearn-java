package org.sklearn.cluster;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class AgglomerativeClusteringTest {

    @Test
    void testBasicClustering() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {1.5, 1.5}, {5.0, 5.0}, {5.5, 5.5}
        });
        AgglomerativeClustering ac = new AgglomerativeClustering(2, "single");
        ac.fit(X);
        assertTrue(ac.isFitted());
        int[] labels = ac.getLabels();
        assertEquals(4, labels.length);
        assertTrue(labels[0] == labels[1]);
        assertTrue(labels[2] == labels[3]);
    }

    @Test
    void testCompleteLinkage() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {8.0, 8.0}, {9.0, 9.0}
        });
        AgglomerativeClustering ac = new AgglomerativeClustering(2, "complete");
        ac.fit(X);
        int[] labels = ac.getLabels();
        assertTrue(labels[0] == labels[1]);
        assertTrue(labels[2] == labels[3]);
    }

    @Test
    void testAverageLinkage() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {1.2, 1.2}, {5.0, 5.0}, {5.2, 5.2}
        });
        AgglomerativeClustering ac = new AgglomerativeClustering(2, "average");
        ac.fit(X);
        int[] labels = ac.getLabels();
        assertTrue(labels[0] == labels[1]);
        assertTrue(labels[2] == labels[3]);
    }

    @Test
    void testWardLinkage() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {10.0, 10.0}, {11.0, 11.0}
        });
        AgglomerativeClustering ac = new AgglomerativeClustering(2, "ward");
        ac.fit(X);
        int[] labels = ac.getLabels();
        assertTrue(labels[0] == labels[1]);
        assertTrue(labels[2] == labels[3]);
    }

    @Test
    void testFitPredict() {
        Matrix X = new Matrix(new double[][]{{0.0, 0.0}, {0.1, 0.1}, {5.0, 5.0}});
        AgglomerativeClustering ac = new AgglomerativeClustering(2, "single");
        int[] labels = ac.fitPredict(X);
        assertEquals(3, labels.length);
    }

    @Test
    void testSingleCluster() {
        Matrix X = new Matrix(new double[][]{{1.0, 1.0}, {2.0, 2.0}});
        AgglomerativeClustering ac = new AgglomerativeClustering(1, "single");
        ac.fit(X);
        int[] labels = ac.getLabels();
        assertEquals(0, labels[0]);
        assertEquals(0, labels[1]);
    }

    @Test
    void testInvalidLinkage() {
        assertThrows(IllegalArgumentException.class,
            () -> new AgglomerativeClustering(2, "unknown"));
    }

    @Test
    void testGetParameters() {
        AgglomerativeClustering ac = new AgglomerativeClustering(3, "complete");
        Map<String, Object> params = ac.getParameters();
        assertEquals(3, params.get("n_clusters"));
        assertEquals("complete", params.get("linkage"));
    }
}
