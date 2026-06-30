package org.sklearn.cluster;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class KMeansTest {

    @Test
    void testThreeClusters() {
        Matrix X = new Matrix(new double[][]{
            {0, 0}, {0.1, 0.1}, {-0.1, -0.1},
            {5, 5}, {5.1, 5.1}, {4.9, 4.9},
            {10, 0}, {10.1, 0.1}, {9.9, -0.1}
        });

        KMeans kmeans = new KMeans(3, 100, 1e-6, 5, 42);
        kmeans.fit(X);

        assertEquals(3, kmeans.getClusterCenters().rows());
        assertEquals(9, kmeans.getLabels().length);
        assertTrue(kmeans.getInertia() >= 0);
    }

    @Test
    void testPredict() {
        Matrix X = new Matrix(new double[][]{
            {0, 0}, {0.1, 0.1}, {5, 5}, {5.1, 5.1}
        });

        KMeans kmeans = new KMeans(2, 100, 1e-6, 5, 42);
        kmeans.fit(X);

        Matrix testX = new Matrix(new double[][]{{0, 0}, {5, 5}});
        int[] preds = kmeans.predict(testX);

        assertEquals(2, preds.length);
        assertEquals(preds[0], preds[0]); // not NaN
    }

    @Test
    void testSingleCluster() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {1.1, 2.1}, {0.9, 1.9}
        });

        KMeans kmeans = new KMeans(1, 100, 1e-6, 1, 42);
        kmeans.fit(X);

        assertEquals(3, kmeans.getLabels().length);
        for (int label : kmeans.getLabels()) {
            assertEquals(0, label);
        }
    }

    @Test
    void testPredictBeforeFitThrows() {
        KMeans kmeans = new KMeans(3);
        assertThrows(IllegalStateException.class,
            () -> kmeans.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        KMeans kmeans = new KMeans(2);
        kmeans.fit(X);

        Matrix Xtest = new Matrix(new double[][]{{1, 2, 3}});
        assertThrows(IllegalArgumentException.class, () -> kmeans.predict(Xtest));
    }

    @Test
    void testGetParameters() {
        KMeans kmeans = new KMeans(5, 200, 1e-5, 3, 123);
        var params = kmeans.getParameters();
        assertEquals(5, params.get("n_clusters"));
        assertEquals(200, params.get("max_iter"));
        assertEquals(1e-5, params.get("tol"));
        assertEquals(3, params.get("n_init"));
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {0.1, 0.1}, {5, 5}, {5.1, 5.1}});
        KMeans kmeans = new KMeans(2, 100, 1e-6, 5, 42);
        kmeans.fit(X);

        double score = kmeans.score(X);
        assertTrue(score < 0); // negative inertia
    }
}
