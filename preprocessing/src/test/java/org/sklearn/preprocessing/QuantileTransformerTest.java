package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class QuantileTransformerTest {

    @Test
    void testBasicFitTransformUniform() {
        Matrix X = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 8.0}, {5.0, 6.0}, {7.0, 4.0}});
        QuantileTransformer qt = new QuantileTransformer(100, "uniform");
        qt.fit(X, null);
        assertTrue(qt.isFitted());
        Matrix Xt = qt.transform(X);
        assertEquals(4, Xt.rows());
        assertEquals(2, Xt.cols());
    }

    @Test
    void testBasicFitTransformNormal() {
        Matrix X = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 8.0}, {5.0, 6.0}});
        QuantileTransformer qt = new QuantileTransformer(100, "normal");
        qt.fit(X, null);
        Matrix Xt = qt.transform(X);
        assertEquals(3, Xt.rows());
        assertEquals(2, Xt.cols());
    }

    @Test
    void testTransformBeforeFitThrows() {
        QuantileTransformer qt = new QuantileTransformer(100, "uniform");
        assertThrows(IllegalStateException.class, () -> qt.transform(new Matrix(1, 1)));
    }

    @Test
    void testFeatureMismatchThrows() {
        QuantileTransformer qt = new QuantileTransformer(100, "uniform");
        qt.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}), null);
        assertThrows(IllegalArgumentException.class,
            () -> qt.transform(new Matrix(new double[][]{{1.0}, {2.0}})));
    }

    @Test
    void testInverseTransformThrows() {
        QuantileTransformer qt = new QuantileTransformer(100, "uniform");
        qt.fit(new Matrix(new double[][]{{1.0}, {2.0}}), null);
        assertThrows(UnsupportedOperationException.class,
            () -> qt.inverseTransform(new Matrix(1, 1)));
    }

    @Test
    void testGetParameters() {
        QuantileTransformer qt = new QuantileTransformer(50, "normal");
        Map<String, Object> params = qt.getParameters();
        assertEquals(50, params.get("n_quantiles"));
        assertEquals("normal", params.get("output_distribution"));
    }
}
