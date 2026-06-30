package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class RobustScalerTest {

    @Test
    void testFitAndTransform() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}, {10.0}, {100.0}});
        RobustScaler rs = new RobustScaler();
        rs.fit(X, null);
        Matrix Xt = rs.transform(X);
        assertEquals(5, Xt.rows());
        assertEquals(1, Xt.cols());
        assertTrue(Xt.get(2, 0) >= -0.5 && Xt.get(2, 0) <= 0.5);
    }

    @Test
    void testInverseTransform() {
        Matrix X = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        RobustScaler rs = new RobustScaler();
        rs.fit(X, null);
        Matrix Xt = rs.transform(X);
        Matrix Xback = rs.inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                assertEquals(X.get(i, j), Xback.get(i, j), 1e-10);
            }
        }
    }

    @Test
    void testTransformBeforeFitThrows() {
        RobustScaler rs = new RobustScaler();
        assertThrows(IllegalStateException.class,
            () -> rs.transform(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        RobustScaler rs = new RobustScaler();
        rs.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}), null);
        assertThrows(IllegalArgumentException.class,
            () -> rs.transform(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testIsFitted() {
        RobustScaler rs = new RobustScaler();
        assertFalse(rs.isFitted());
        rs.fit(new Matrix(new double[][]{{1.0}, {2.0}}), null);
        assertTrue(rs.isFitted());
    }

    @Test
    void testGetParameters() {
        RobustScaler rs = new RobustScaler();
        rs.fit(new Matrix(new double[][]{{1.0}, {2.0}}), null);
        var params = rs.getParameters();
        assertTrue(params.containsKey("center"));
        assertTrue(params.containsKey("scale"));
    }
}
