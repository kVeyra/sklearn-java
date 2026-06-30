package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class StandardScalerTest {

    @Test
    void testBasicFitTransform() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        });

        StandardScaler scaler = new StandardScaler();
        scaler.fit(X, null);

        assertTrue(scaler.isFitted());

        Map<String, Object> params = scaler.getParameters();
        Vector mean = (Vector) params.get("mean");
        Vector scale = (Vector) params.get("scale");
        Vector var = (Vector) params.get("var");

        assertEquals(3.0, mean.get(0), 1e-12);
        assertEquals(4.0, mean.get(1), 1e-12);
        assertEquals(8.0 / 3.0, var.get(0), 1e-12);
        assertEquals(8.0 / 3.0, var.get(1), 1e-12);
        assertEquals(Math.sqrt(8.0 / 3.0), scale.get(0), 1e-12);
        assertEquals(Math.sqrt(8.0 / 3.0), scale.get(1), 1e-12);
        assertEquals(3, params.get("n_samples_seen"));

        Matrix Xt = scaler.transform(X);
        assertEquals(3, Xt.rows());
        assertEquals(2, Xt.cols());
    }

    @Test
    void testTransformedMeanAndStd() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {10.0, 11.0, 12.0}
        });

        StandardScaler scaler = new StandardScaler();
        scaler.fit(X, null);
        Matrix Xt = scaler.transform(X);

        for (int j = 0; j < Xt.cols(); j++) {
            double sum = 0.0;
            for (int i = 0; i < Xt.rows(); i++) {
                sum += Xt.get(i, j);
            }
            double colMean = sum / Xt.rows();
            assertEquals(0.0, colMean, 1e-12);
        }

        for (int j = 0; j < Xt.cols(); j++) {
            double sumSq = 0.0;
            for (int i = 0; i < Xt.rows(); i++) {
                sumSq += Xt.get(i, j) * Xt.get(i, j);
            }
            double colVar = sumSq / Xt.rows();
            assertEquals(1.0, colVar, 1e-12);
        }
    }

    @Test
    void testInverseTransformRoundtrip() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        });

        StandardScaler scaler = new StandardScaler();
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
    void testZeroVarianceFeature() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 5.0},
            {2.0, 5.0},
            {3.0, 5.0}
        });

        StandardScaler scaler = new StandardScaler();
        scaler.fit(X, null);

        Vector scale = (Vector) scaler.getParameters().get("scale");
        Vector var = (Vector) scaler.getParameters().get("var");

        assertEquals(2.0 / 3.0, var.get(0), 1e-12);
        assertEquals(0.0, var.get(1), 1e-12);
        assertEquals(1.0, scale.get(1), 1e-12);

        Matrix Xt = scaler.transform(X);
        assertEquals(0.0, Xt.get(0, 1), 1e-12);
        assertEquals(0.0, Xt.get(1, 1), 1e-12);
        assertEquals(0.0, Xt.get(2, 1), 1e-12);

        Matrix Xr = scaler.inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            assertEquals(X.get(i, 0), Xr.get(i, 0), 1e-12);
            assertEquals(X.get(i, 1), Xr.get(i, 1), 1e-12);
        }
    }

    @Test
    void testSingleSample() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0, 3.0}
        });

        StandardScaler scaler = new StandardScaler();
        scaler.fit(X, null);

        Vector scale = (Vector) scaler.getParameters().get("scale");
        for (int j = 0; j < scale.size(); j++) {
            assertEquals(1.0, scale.get(j), 1e-12);
        }

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

        StandardScaler scaler = new StandardScaler();
        scaler.fit(X, null);
        Matrix Xt = scaler.transform(X);

        assertNotSame(X, Xt);

        Xt.set(0, 0, 999.0);
        assertEquals(1.0, X.get(0, 0), 1e-12);
    }

    @Test
    void testFitTransformConvenience() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        });

        StandardScaler scaler = new StandardScaler();
        Matrix Xt = scaler.fitTransform(X, null);

        assertTrue(scaler.isFitted());
        assertEquals(3, Xt.rows());
        assertEquals(2, Xt.cols());
    }

    @Test
    void testGetParameters() {
        Matrix X = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        StandardScaler scaler = new StandardScaler();
        scaler.fit(X, null);

        Map<String, Object> params = scaler.getParameters();
        assertTrue(params.containsKey("mean"));
        assertTrue(params.containsKey("scale"));
        assertTrue(params.containsKey("var"));
        assertTrue(params.containsKey("n_samples_seen"));
        assertEquals(2, params.get("n_samples_seen"));
    }

    @Test
    void testTransformFailsWhenNotFitted() {
        StandardScaler scaler = new StandardScaler();
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}});
        assertThrows(IllegalStateException.class, () -> scaler.transform(X));
    }

    @Test
    void testInverseTransformFailsWhenNotFitted() {
        StandardScaler scaler = new StandardScaler();
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}});
        assertThrows(IllegalStateException.class, () -> scaler.inverseTransform(X));
    }

    @Test
    void testTransformFailsOnWrongFeatureCount() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        });

        StandardScaler scaler = new StandardScaler();
        scaler.fit(X, null);

        Matrix wrongX = new Matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0}
        });

        assertThrows(IllegalArgumentException.class, () -> scaler.transform(wrongX));
    }

    @Test
    void testDeterministic() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        });

        StandardScaler s1 = new StandardScaler();
        s1.fit(X, null);

        StandardScaler s2 = new StandardScaler();
        s2.fit(X, null);

        Map<String, Object> p1 = s1.getParameters();
        Map<String, Object> p2 = s2.getParameters();

        assertEquals(p1.get("mean"), p2.get("mean"));
        assertEquals(p1.get("scale"), p2.get("scale"));
        assertEquals(p1.get("var"), p2.get("var"));

        Matrix t1 = s1.transform(X);
        Matrix t2 = s2.transform(X);
        assertEquals(t1, t2);
    }
}
