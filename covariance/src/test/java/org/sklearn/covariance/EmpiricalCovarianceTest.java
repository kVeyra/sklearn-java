package org.sklearn.covariance;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class EmpiricalCovarianceTest {

    @Test
    void testBasicFit() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}
        });
        EmpiricalCovariance ec = new EmpiricalCovariance(false);
        ec.fit(X, null);
        assertTrue(ec.isFitted());
        Matrix cov = ec.getCovariance();
        assertEquals(2, cov.rows());
        assertEquals(2, cov.cols());
        Matrix prec = ec.getPrecision();
        assertEquals(2, prec.rows());
        assertEquals(2, prec.cols());
    }

    @Test
    void testWithAssumeCentered() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0}, {-1.0, -2.0}, {0.0, 0.0}
        });
        EmpiricalCovariance ec = new EmpiricalCovariance(true);
        ec.fit(X, null);
        Matrix cov = ec.getCovariance();
        assertEquals(2, cov.rows());
        assertEquals(2, cov.cols());
    }

    @Test
    void testGetLocation() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0}, {3.0, 4.0}
        });
        EmpiricalCovariance ec = new EmpiricalCovariance(false);
        ec.fit(X, null);
        Vector loc = ec.getMean();
        assertEquals(2, loc.size());
        assertEquals(2.0, loc.get(0), 1e-12);
        assertEquals(3.0, loc.get(1), 1e-12);
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
        EmpiricalCovariance ec = new EmpiricalCovariance(false);
        ec.fit(X, null);
        Vector mah = ec.mahalanobis(X);
        assertEquals(n, mah.size());
    }

    @Test
    void testFitBeforeGetPrecisionThrows() {
        EmpiricalCovariance ec = new EmpiricalCovariance(false);
        assertThrows(IllegalStateException.class, ec::getPrecision);
    }

    @Test
    void testGetParameters() {
        EmpiricalCovariance ec = new EmpiricalCovariance(true);
        ec.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}), null);
        Map<String, Object> params = ec.getParameters();
        assertNotNull(params.get("covariance_"));
        assertNotNull(params.get("precision_"));
    }
}
