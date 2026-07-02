package org.sklearn.covariance;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class LedoitWolfTest {

    @Test
    void testBasicFit() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}
        });
        LedoitWolf lw = new LedoitWolf(false);
        lw.fit(X, null);
        assertTrue(lw.isFitted());
        Matrix cov = lw.getCovariance();
        assertEquals(2, cov.rows());
        assertEquals(2, cov.cols());
    }

    @Test
    void testShrinkage() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}
        });
        LedoitWolf lw = new LedoitWolf(false);
        lw.fit(X, null);
        double shrinkage = lw.getShrinkage();
        assertTrue(shrinkage >= 0.0);
        assertTrue(shrinkage <= 1.0);
    }

    @Test
    void testWithAssumeCentered() {
        Matrix X = new Matrix(new double[][]{
            {0.0, 0.0}, {1.0, 2.0}, {-1.0, -2.0}
        });
        LedoitWolf lw = new LedoitWolf(true);
        lw.fit(X, null);
        Matrix cov = lw.getCovariance();
        assertEquals(2, cov.rows());
        assertEquals(2, cov.cols());
    }

    @Test
    void testGetLocation() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0}, {3.0, 4.0}
        });
        LedoitWolf lw = new LedoitWolf(false);
        lw.fit(X, null);
        Vector loc = lw.getMean();
        assertEquals(2, loc.size());
    }

    @Test
    void testMahalanobis() {
        int n = 20;
        Matrix X = new Matrix(n, 3);
        for (int i = 0; i < n; i++) {
            X.set(i, 0, Math.sin(i * 0.5));
            X.set(i, 1, Math.cos(i * 0.3));
            X.set(i, 2, i * 0.1);
        }
        LedoitWolf lw = new LedoitWolf(false);
        lw.fit(X, null);
        Vector mah = lw.mahalanobis(X);
        assertEquals(n, mah.size());
    }

    @Test
    void testFitBeforeGetPrecisionThrows() {
        LedoitWolf lw = new LedoitWolf(false);
        assertThrows(IllegalStateException.class, lw::getPrecision);
    }

    @Test
    void testGetParameters() {
        LedoitWolf lw = new LedoitWolf(true);
        lw.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}), null);
        Map<String, Object> params = lw.getParameters();
        assertNotNull(params.get("shrinkage"));
        assertNotNull(params.get("covariance_"));
    }
}
