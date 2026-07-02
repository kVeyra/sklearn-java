package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class OneHotEncoderTest {

    @Test
    void testBasicOneHotEncoding() {
        Matrix X = new Matrix(new double[][]{
            {0, 10},
            {1, 20},
            {2, 30}
        });

        OneHotEncoder enc = new OneHotEncoder();
        enc.fit(X, null);

        assertTrue(enc.isFitted());

        Matrix Xt = enc.transform(X);
        assertEquals(3, Xt.rows());
        assertEquals(6, Xt.cols());

        double[] row0 = {1, 0, 0, 1, 0, 0};
        double[] row1 = {0, 1, 0, 0, 1, 0};
        double[] row2 = {0, 0, 1, 0, 0, 1};
        assertRowEquals(Xt, 0, row0);
        assertRowEquals(Xt, 1, row1);
        assertRowEquals(Xt, 2, row2);
    }

    @Test
    void testFitTransformRoundtrip() {
        Matrix X = new Matrix(new double[][]{
            {0, 10},
            {1, 20},
            {2, 30},
            {0, 30},
            {2, 10}
        });

        OneHotEncoder enc = new OneHotEncoder();
        Matrix Xt = enc.fitTransform(X, null);

        assertTrue(enc.isFitted());

        Matrix Xr = enc.inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                assertEquals(X.get(i, j), Xr.get(i, j), 1e-12);
            }
        }
    }

    @Test
    void testDropFirstCategory() {
        Matrix X = new Matrix(new double[][]{
            {0, 10},
            {1, 20},
            {2, 30}
        });

        OneHotEncoder enc = new OneHotEncoder(true);
        enc.fit(X, null);

        Matrix Xt = enc.transform(X);
        assertEquals(3, Xt.rows());
        assertEquals(4, Xt.cols());

        double[] row0 = {0, 0, 0, 0};
        double[] row1 = {1, 0, 1, 0};
        double[] row2 = {0, 1, 0, 1};
        assertRowEquals(Xt, 0, row0);
        assertRowEquals(Xt, 1, row1);
        assertRowEquals(Xt, 2, row2);

        Matrix Xr = enc.inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                assertEquals(X.get(i, j), Xr.get(i, j), 1e-12);
            }
        }
    }

    @Test
    void testSingleFeatureBinaryCategories() {
        Matrix X = new Matrix(new double[][]{
            {0},
            {1},
            {0}
        });

        OneHotEncoder enc = new OneHotEncoder();
        enc.fit(X, null);

        Matrix Xt = enc.transform(X);
        assertEquals(3, Xt.rows());
        assertEquals(2, Xt.cols());

        assertEquals(1.0, Xt.get(0, 0), 1e-12);
        assertEquals(0.0, Xt.get(0, 1), 1e-12);
        assertEquals(0.0, Xt.get(1, 0), 1e-12);
        assertEquals(1.0, Xt.get(1, 1), 1e-12);
        assertEquals(1.0, Xt.get(2, 0), 1e-12);
        assertEquals(0.0, Xt.get(2, 1), 1e-12);

        Matrix Xr = enc.inverseTransform(Xt);
        for (int i = 0; i < X.rows(); i++) {
            assertEquals(X.get(i, 0), Xr.get(i, 0), 1e-12);
        }
    }

    @Test
    void testSingleSample() {
        Matrix X = new Matrix(new double[][]{
            {5, 9}
        });

        OneHotEncoder enc = new OneHotEncoder();
        enc.fit(X, null);

        assertTrue(enc.isFitted());

        Matrix Xt = enc.transform(X);
        assertEquals(1, Xt.rows());
        assertEquals(2, Xt.cols());

        assertEquals(1.0, Xt.get(0, 0), 1e-12);
        assertEquals(1.0, Xt.get(0, 1), 1e-12);

        Matrix Xr = enc.inverseTransform(Xt);
        assertEquals(X.get(0, 0), Xr.get(0, 0), 1e-12);
        assertEquals(X.get(0, 1), Xr.get(0, 1), 1e-12);
    }

    @Test
    void testTransformFailsWhenNotFitted() {
        OneHotEncoder enc = new OneHotEncoder();
        Matrix X = new Matrix(new double[][]{{0}, {1}});
        assertThrows(IllegalStateException.class, () -> enc.transform(X));
    }

    @Test
    void testInverseTransformFailsWhenNotFitted() {
        OneHotEncoder enc = new OneHotEncoder();
        Matrix X = new Matrix(new double[][]{{1, 0}});
        assertThrows(IllegalStateException.class, () -> enc.inverseTransform(X));
    }

    @Test
    void testTransformFailsOnUnknownCategory() {
        Matrix X = new Matrix(new double[][]{
            {0, 10},
            {1, 20}
        });

        OneHotEncoder enc = new OneHotEncoder();
        enc.fit(X, null);

        Matrix badX = new Matrix(new double[][]{{99, 10}});
        assertThrows(IllegalArgumentException.class, () -> enc.transform(badX));
    }

    @Test
    void testTransformFailsOnWrongFeatureCount() {
        Matrix X = new Matrix(new double[][]{
            {0, 10},
            {1, 20}
        });

        OneHotEncoder enc = new OneHotEncoder();
        enc.fit(X, null);

        Matrix badX = new Matrix(new double[][]{{0}});
        assertThrows(IllegalArgumentException.class, () -> enc.transform(badX));
    }

    @Test
    void testGetParameters() {
        Matrix X = new Matrix(new double[][]{
            {0, 10},
            {1, 20},
            {2, 30}
        });

        OneHotEncoder enc = new OneHotEncoder();
        enc.fit(X, null);

        Map<String, Object> params = enc.getParameters();
        assertTrue(params.containsKey("categories"));

        @SuppressWarnings("unchecked")
        List<int[]> cats = (List<int[]>) params.get("categories");
        assertEquals(2, cats.size());
        assertArrayEquals(new int[]{0, 1, 2}, cats.get(0));
        assertArrayEquals(new int[]{10, 20, 30}, cats.get(1));
    }

    private static void assertRowEquals(Matrix m, int row, double[] expected) {
        for (int j = 0; j < expected.length; j++) {
            assertEquals(expected[j], m.get(row, j), 1e-12,
                "Mismatch at row " + row + ", col " + j);
        }
    }
}
