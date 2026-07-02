package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class PowerTransformerTest {

    @Test
    void testBasicFitTransformBoxCox() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}});
        PowerTransformer pt = new PowerTransformer("box-cox", false);
        pt.fit(X, null);
        assertTrue(pt.isFitted());
        Matrix Xt = pt.transform(X);
        assertEquals(5, Xt.rows());
        assertEquals(1, Xt.cols());
        assertNotNull(pt.getLambdas());
        assertEquals(1, pt.getLambdas().length);
    }

    @Test
    void testBasicFitTransformYeoJohnson() {
        Matrix X = new Matrix(new double[][]{{-2.0}, {-1.0}, {0.0}, {1.0}, {2.0}});
        PowerTransformer pt = new PowerTransformer("yeo-johnson", false);
        pt.fit(X, null);
        Matrix Xt = pt.transform(X);
        assertEquals(5, Xt.rows());
    }

    @Test
    void testWithStandardize() {
        Matrix X = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        PowerTransformer pt = new PowerTransformer("yeo-johnson", true);
        pt.fit(X, null);
        Matrix Xt = pt.transform(X);
        assertEquals(3, Xt.rows());
        assertEquals(2, Xt.cols());
    }

    @Test
    void testInverseTransformRoundtrip() {
        Matrix X = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        PowerTransformer pt = new PowerTransformer("yeo-johnson", false);
        pt.fit(X, null);
        Matrix Xt = pt.transform(X);
        Matrix Xinv = pt.inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                assertEquals(X.get(i, j), Xinv.get(i, j), 1e-8);
            }
        }
    }

    @Test
    void testTransformBeforeFitThrows() {
        PowerTransformer pt = new PowerTransformer("box-cox", false);
        Matrix X = new Matrix(new double[][]{{1.0, 2.0}});
        assertThrows(IllegalStateException.class, () -> pt.transform(X));
    }

    @Test
    void testFeatureMismatchThrows() {
        PowerTransformer pt = new PowerTransformer("box-cox", false);
        pt.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}), null);
        Matrix Xbad = new Matrix(new double[][]{{1.0}, {2.0}});
        assertThrows(IllegalArgumentException.class, () -> pt.transform(Xbad));
    }

    @Test
    void testGetParameters() {
        PowerTransformer pt = new PowerTransformer("box-cox", true);
        Map<String, Object> params = pt.getParameters();
        assertEquals("box-cox", params.get("method"));
    }

    @Test
    void testBoxCoxRequiresPositive() {
        PowerTransformer pt = new PowerTransformer("box-cox", false);
        Matrix X = new Matrix(new double[][]{{-1.0}, {1.0}});
        pt.fit(X, null);
        assertThrows(IllegalArgumentException.class, () -> pt.transform(X));
    }
}
