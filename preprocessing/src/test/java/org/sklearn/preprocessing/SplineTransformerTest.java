package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SplineTransformerTest {

    @Test
    void testBasicFitTransform() {
        Matrix X = new Matrix(new double[][]{{0.0}, {0.5}, {1.0}});
        SplineTransformer st = new SplineTransformer(5, 3);
        st.fit(X, null);
        assertTrue(st.isFitted());
        Matrix Xt = st.transform(X);
        assertEquals(3, Xt.rows());
        int expectedCols = (5 + 3 - 1); // nKnots + degree - 1
        assertEquals(expectedCols, Xt.cols());
    }

    @Test
    void testLinearSpline() {
        Matrix X = new Matrix(new double[][]{{0.0}, {0.5}, {1.0}});
        SplineTransformer st = new SplineTransformer(3, 1);
        st.fit(X, null);
        Matrix Xt = st.transform(X);
        assertEquals(3, Xt.rows());
        assertEquals(3 + 1 - 1, Xt.cols());
    }

    @Test
    void testMultiFeature() {
        Matrix X = new Matrix(new double[][]{{0.0, 0.0}, {0.5, 1.0}, {1.0, 0.0}});
        SplineTransformer st = new SplineTransformer(4, 2);
        st.fit(X, null);
        Matrix Xt = st.transform(X);
        assertEquals(3, Xt.rows());
    }

    @Test
    void testTransformBeforeFitThrows() {
        SplineTransformer st = new SplineTransformer(5, 3);
        assertThrows(IllegalStateException.class, () -> st.transform(new Matrix(1, 1)));
    }

    @Test
    void testFeatureMismatchThrows() {
        SplineTransformer st = new SplineTransformer(5, 3);
        st.fit(new Matrix(new double[][]{{0.0, 1.0}, {0.5, 2.0}}), null);
        assertThrows(IllegalArgumentException.class,
            () -> st.transform(new Matrix(new double[][]{{0.0}, {0.5}})));
    }

    @Test
    void testInverseTransformThrows() {
        SplineTransformer st = new SplineTransformer(5, 3);
        st.fit(new Matrix(new double[][]{{0.0}, {1.0}}), null);
        assertThrows(UnsupportedOperationException.class,
            () -> st.inverseTransform(new Matrix(1, 1)));
    }

    @Test
    void testInvalidConstructor() {
        assertThrows(IllegalArgumentException.class, () -> new SplineTransformer(1, 3));
        assertThrows(IllegalArgumentException.class, () -> new SplineTransformer(5, 0));
    }

    @Test
    void testGetParameters() {
        SplineTransformer st = new SplineTransformer(10, 2);
        Map<String, Object> params = st.getParameters();
        assertEquals(10, params.get("n_knots"));
        assertEquals(2, params.get("degree"));
    }
}
