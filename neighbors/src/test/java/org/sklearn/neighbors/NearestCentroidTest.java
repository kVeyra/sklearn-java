package org.sklearn.neighbors;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class NearestCentroidTest {

    @Test
    void testFitPredict() {
        Matrix X = new Matrix(new double[][]{
            {1, 1},
            {2, 2},
            {10, 10},
            {11, 11}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        NearestCentroid nc = new NearestCentroid();
        nc.fit(X, y);
        Vector preds = nc.predict(X);
        assertEquals(0.0, preds.get(0));
        assertEquals(1.0, preds.get(2));
    }

    @Test
    void testCentroids() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}});
        Vector y = new Vector(new double[]{0, 1, 1});

        NearestCentroid nc = new NearestCentroid();
        nc.fit(X, y);
        assertNotNull(nc.getCentroids());
        assertEquals(2, nc.getCentroids().size());
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{{1, 1}, {2, 2}, {10, 10}, {11, 11}});
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        NearestCentroid nc = new NearestCentroid();
        nc.fit(X, y);
        assertEquals(1.0, nc.score(X, y));
    }
}
