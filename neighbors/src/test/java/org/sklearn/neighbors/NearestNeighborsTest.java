package org.sklearn.neighbors;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import static org.junit.jupiter.api.Assertions.*;

class NearestNeighborsTest {

    @Test
    void testKneighbors() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}, {2, 2}, {10, 10}});
        NearestNeighbors nn = new NearestNeighbors(2, 1.0);
        nn.fit(X);
        int[][] idx = nn.kneighbors(new Matrix(new double[][]{{0, 0}}));
        assertEquals(1, idx.length);
        assertEquals(2, idx[0].length);
        assertEquals(0, idx[0][0]);
    }

    @Test
    void testRadiusNeighbors() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}, {5, 5}});
        NearestNeighbors nn = new NearestNeighbors(2, 1.5);
        nn.fit(X);
        var result = nn.radiusNeighbors(new Matrix(new double[][]{{0, 0}}));
        assertEquals(1, result.size());
    }

    @Test
    void testKneighborsDistances() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}, {5, 5}});
        NearestNeighbors nn = new NearestNeighbors(2, 1.0);
        nn.fit(X);
        Matrix dist = nn.kneighborsDistances(new Matrix(new double[][]{{0, 0}}));
        assertEquals(1, dist.rows());
        assertEquals(2, dist.cols());
    }

    @Test
    void testFitBeforePredict() {
        NearestNeighbors nn = new NearestNeighbors();
        assertThrows(IllegalStateException.class, () -> nn.kneighbors(new Matrix(1, 1)));
    }
}
