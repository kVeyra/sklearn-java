package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class MinMaxScalerTest {

    @Test
    void testBasicFitTransform() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        });

        MinMaxScaler scaler = new MinMaxScaler();
        scaler.fit(X, null);

        assertTrue(scaler.isFitted());

        Matrix Xt = scaler.transform(X);
        assertEquals(3, Xt.rows());
        assertEquals(2, Xt.cols());

        for (int j = 0; j < Xt.cols(); j++) {
            assertEquals(0.0, Xt.get(0, j), 1e-12);
            assertEquals(0.5, Xt.get(1, j), 1e-12);
            assertEquals(1.0, Xt.get(2, j), 1e-12);
        }
    }

    @Test
    void testCustomFeatureRange() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 10.0},
            {3.0, 20.0},
            {5.0, 30.0}
        });

        MinMaxScaler scaler = new MinMaxScaler(1.0, 2.0);
        scaler.fit(X, null);

        Matrix Xt = scaler.transform(X);
        assertEquals(1.0, Xt.get(0, 0), 1e-12);
        assertEquals(1.5, Xt.get(1, 0), 1e-12);
        assertEquals(2.0, Xt.get(2, 0), 1e-12);
        assertEquals(1.0, Xt.get(0, 1), 1e-12);
        assertEquals(1.5, Xt.get(1, 1), 1e-12);
        assertEquals(2.0, Xt.get(2, 1), 1e-12);
    }

    @Test
    void testInverseTransformRoundtrip() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        });

        MinMaxScaler scaler = new MinMaxScaler();
        scaler.fit(X, null);

        Matrix Xt = scaler.transform(X);
        Matrix Xr = scaler.inverseTransform(Xt);

        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                assertEquals(X.get(i, j), Xr.get(i, j), 1e-12);
            }
        }
    }

    @Test
    void testInverseTransformRoundtripCustomRange() {
        Matrix X = new Matrix(new double[][]{
            {10.0, 100.0},
            {20.0, 200.0},
            {30.0, 300.0}
        });

        MinMaxScaler scaler = new MinMaxScaler(-1.0, 1.0);
        scaler.fit(X, null);

        Matrix Xt = scaler.transform(X);
        Matrix Xr = scaler.inverseTransform(Xt);

        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                assertEquals(X.get(i, j), Xr.get(i, j), 1e-12);
            }
        }
    }

    @Test
    void testSingleFeature() {
        Matrix X = new Matrix(new double[][]{
            {5.0},
            {10.0},
            {15.0}
        });

        MinMaxScaler scaler = new MinMaxScaler();
        scaler.fit(X, null);

        Matrix Xt = scaler.transform(X);
        assertEquals(0.0, Xt.get(0, 0), 1e-12);
        assertEquals(0.5, Xt.get(1, 0), 1e-12);
        assertEquals(1.0, Xt.get(2, 0), 1e-12);

        Matrix Xr = scaler.inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            assertEquals(X.get(i, 0), Xr.get(i, 0), 1e-12);
        }
    }

    @Test
    void testConstantFeature() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 5.0},
            {2.0, 5.0},
            {3.0, 5.0}
        });

        MinMaxScaler scaler = new MinMaxScaler();
        scaler.fit(X, null);

        Vector scale = (Vector) scaler.getParameters().get("scale");
        assertEquals(2.0, scale.get(0), 1e-12);
        assertEquals(1.0, scale.get(1), 1e-12);

        Matrix Xt = scaler.transform(X);
        for (int i = 0; i < Xt.rows(); i++) {
            assertEquals(0.0, Xt.get(i, 1), 1e-12);
        }

        Matrix Xr = scaler.inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            assertEquals(X.get(i, 0), Xr.get(i, 0), 1e-12);
            assertEquals(X.get(i, 1), Xr.get(i, 1), 1e-12);
        }
    }

    @Test
    void testAllConstantFeatures() {
        Matrix X = new Matrix(new double[][]{
            {7.0, 7.0},
            {7.0, 7.0},
            {7.0, 7.0}
        });

        MinMaxScaler scaler = new MinMaxScaler();
        scaler.fit(X, null);

        Vector scale = (Vector) scaler.getParameters().get("scale");
        assertEquals(1.0, scale.get(0), 1e-12);
        assertEquals(1.0, scale.get(1), 1e-12);

        Matrix Xt = scaler.transform(X);
        for (int i = 0; i < Xt.rows(); i++) {
            for (int j = 0; j < Xt.cols(); j++) {
                assertEquals(0.0, Xt.get(i, j), 1e-12);
            }
        }

        Matrix Xr = scaler.inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                assertEquals(X.get(i, j), Xr.get(i, j), 1e-12);
            }
        }
    }

    @Test
    void testTransformReturnsNewCopy() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0}
        });

        MinMaxScaler scaler = new MinMaxScaler();
        scaler.fit(X, null);
        Matrix Xt = scaler.transform(X);

        assertNotSame(X, Xt);

        Xt.set(0, 0, 999.0);
        assertEquals(1.0, X.get(0, 0), 1e-12);
    }

    @Test
    void testInverseTransformReturnsNewCopy() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0}
        });

        MinMaxScaler scaler = new MinMaxScaler();
        scaler.fit(X, null);
        Matrix Xt = scaler.transform(X);
        Matrix Xr = scaler.inverseTransform(Xt);

        assertNotSame(Xt, Xr);

        Xr.set(0, 0, 999.0);
        assertEquals(0.0, Xt.get(0, 0), 1e-12);
    }

    @Test
    void testGetParameters() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 10.0},
            {3.0, 20.0},
            {5.0, 30.0}
        });

        MinMaxScaler scaler = new MinMaxScaler();
        scaler.fit(X, null);

        Map<String, Object> params = scaler.getParameters();
        assertTrue(params.containsKey("data_min"));
        assertTrue(params.containsKey("data_max"));
        assertTrue(params.containsKey("data_range"));
        assertTrue(params.containsKey("scale"));
        assertTrue(params.containsKey("min"));
        assertTrue(params.containsKey("feature_range"));

        Vector dataMin = (Vector) params.get("data_min");
        Vector dataMax = (Vector) params.get("data_max");
        Vector dataRange = (Vector) params.get("data_range");
        Vector scale = (Vector) params.get("scale");
        Vector min = (Vector) params.get("min");
        double[] fRange = (double[]) params.get("feature_range");

        assertEquals(1.0, dataMin.get(0), 1e-12);
        assertEquals(10.0, dataMin.get(1), 1e-12);
        assertEquals(5.0, dataMax.get(0), 1e-12);
        assertEquals(30.0, dataMax.get(1), 1e-12);
        assertEquals(4.0, dataRange.get(0), 1e-12);
        assertEquals(20.0, dataRange.get(1), 1e-12);
        assertEquals(4.0, scale.get(0), 1e-12);
        assertEquals(20.0, scale.get(1), 1e-12);
        assertEquals(0.0, fRange[0], 1e-12);
        assertEquals(1.0, fRange[1], 1e-12);
    }

    @Test
    void testConstructorThrowsOnInvalidRange() {
        assertThrows(IllegalArgumentException.class, () -> new MinMaxScaler(2.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new MinMaxScaler(1.0, 1.0));
    }

    @Test
    void testTransformFailsWhenNotFitted() {
        MinMaxScaler scaler = new MinMaxScaler();
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}});
        assertThrows(IllegalStateException.class, () -> scaler.transform(X));
    }

    @Test
    void testInverseTransformFailsWhenNotFitted() {
        MinMaxScaler scaler = new MinMaxScaler();
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}});
        assertThrows(IllegalStateException.class, () -> scaler.inverseTransform(X));
    }

    @Test
    void testTransformFailsOnWrongFeatureCount() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        });

        MinMaxScaler scaler = new MinMaxScaler();
        scaler.fit(X, null);

        Matrix wrongX = new Matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0}
        });

        assertThrows(IllegalArgumentException.class, () -> scaler.transform(wrongX));
    }

    @Test
    void testFitTransformConvenience() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        });

        MinMaxScaler scaler = new MinMaxScaler();
        Matrix Xt = scaler.fitTransform(X, null);

        assertTrue(scaler.isFitted());
        assertEquals(3, Xt.rows());
        assertEquals(2, Xt.cols());
    }

    @Test
    void testDeterministic() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        });

        MinMaxScaler s1 = new MinMaxScaler();
        s1.fit(X, null);

        MinMaxScaler s2 = new MinMaxScaler();
        s2.fit(X, null);

        Map<String, Object> p1 = s1.getParameters();
        Map<String, Object> p2 = s2.getParameters();

        assertEquals(p1.get("data_min"), p2.get("data_min"));
        assertEquals(p1.get("data_max"), p2.get("data_max"));
        assertEquals(p1.get("scale"), p2.get("scale"));

        Matrix t1 = s1.transform(X);
        Matrix t2 = s2.transform(X);
        assertEquals(t1, t2);
    }
}
