package org.sklearn.decomposition;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class PCATest {

    @Test
    void testPCAReducesDimensionCorrectly() {
        Matrix X = new Matrix(new double[][]{
            {2.5, 2.4},
            {0.5, 0.7},
            {2.2, 2.9},
            {1.9, 2.2},
            {3.1, 3.0},
            {2.3, 2.7},
            {2.0, 1.6},
            {1.0, 1.1},
            {1.5, 1.6},
            {1.1, 0.9}
        });

        PCA pca = new PCA(1);
        pca.fit(X, null);

        assertEquals(1, pca.getNComponents());
        assertTrue(pca.isFitted());

        Matrix Xt = pca.transform(X);
        assertEquals(10, Xt.rows());
        assertEquals(1, Xt.cols());
    }

    @Test
    void testExplainedVarianceRatioSumsToOne() {
        Matrix X = new Matrix(new double[][]{
            {2.5, 2.4, 3.0},
            {0.5, 0.7, 1.0},
            {2.2, 2.9, 2.5},
            {1.9, 2.2, 2.0},
            {3.1, 3.0, 3.5}
        });

        PCA pca = new PCA(3);
        pca.fit(X, null);

        Map<String, Object> params = pca.getParameters();
        Vector ratio = (Vector) params.get("explained_variance_ratio");

        assertEquals(3, ratio.size());
        double sum = ratio.sum();
        assertEquals(1.0, sum, 1e-10);
    }

    @Test
    void testTransformInverseTransformRoundtripLossless() {
        Matrix X = new Matrix(new double[][]{
            {2.5, 2.4, 3.0, 1.5},
            {0.5, 0.7, 1.0, 2.0},
            {2.2, 2.9, 2.5, 3.2},
            {1.9, 2.2, 2.0, 1.8},
            {3.1, 3.0, 3.5, 2.9},
            {2.3, 2.7, 2.1, 2.2}
        });

        PCA pca = new PCA(4);
        pca.fit(X, null);

        Matrix Xt = pca.transform(X);
        Matrix Xr = pca.inverseTransform(Xt);

        assertEquals(X.rows(), Xr.rows());
        assertEquals(X.cols(), Xr.cols());
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                assertEquals(X.get(i, j), Xr.get(i, j), 1e-10);
            }
        }
    }

    @Test
    void testTransformInverseTransformRoundtripLossy() {
        Matrix X = new Matrix(new double[][]{
            {2.5, 2.4, 3.0, 1.5},
            {0.5, 0.7, 1.0, 2.0},
            {2.2, 2.9, 2.5, 3.2},
            {1.9, 2.2, 2.0, 1.8},
            {3.1, 3.0, 3.5, 2.9},
            {2.3, 2.7, 2.1, 2.2}
        });

        PCA pca = new PCA(2);
        pca.fit(X, null);

        Matrix Xt = pca.transform(X);
        Matrix Xr = pca.inverseTransform(Xt);

        assertEquals(X.rows(), Xr.rows());
        assertEquals(X.cols(), Xr.cols());
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                assertEquals(X.get(i, j), Xr.get(i, j), 0.5);
            }
        }
    }

    @Test
    void testComponentsAreOrthonormal() {
        Matrix X = new Matrix(new double[][]{
            {2.5, 2.4, 3.0, 1.5, 2.0},
            {0.5, 0.7, 1.0, 2.0, 1.0},
            {2.2, 2.9, 2.5, 3.2, 3.0},
            {1.9, 2.2, 2.0, 1.8, 0.5},
            {3.1, 3.0, 3.5, 2.9, 2.5},
            {2.3, 2.7, 2.1, 2.2, 1.8},
            {1.0, 1.5, 1.2, 1.9, 2.2}
        });

        int k = 3;
        PCA pca = new PCA(k);
        pca.fit(X, null);

        Map<String, Object> params = pca.getParameters();
        Matrix components = (Matrix) params.get("components");

        assertEquals(k, components.rows());
        assertEquals(5, components.cols());

        for (int i = 0; i < k; i++) {
            Vector vi = components.row(i);
            double norm = vi.norm();
            assertEquals(1.0, norm, 1e-10);
            for (int j = i + 1; j < k; j++) {
                Vector vj = components.row(j);
                double dot = vi.dot(vj);
                assertEquals(0.0, dot, 1e-10);
            }
        }
    }

    @Test
    void testGetParametersReturnsCorrectValues() {
        Matrix X = new Matrix(new double[][]{
            {2.5, 2.4},
            {0.5, 0.7},
            {2.2, 2.9}
        });

        PCA pca = new PCA(1);
        pca.fit(X, null);

        Map<String, Object> params = pca.getParameters();
        assertTrue(params.containsKey("n_components"));
        assertTrue(params.containsKey("components"));
        assertTrue(params.containsKey("explained_variance"));
        assertTrue(params.containsKey("explained_variance_ratio"));
        assertTrue(params.containsKey("singular_values"));
        assertTrue(params.containsKey("mean"));

        assertEquals(1, params.get("n_components"));

        Matrix components = (Matrix) params.get("components");
        assertEquals(1, components.rows());
        assertEquals(2, components.cols());

        Vector explainedVariance = (Vector) params.get("explained_variance");
        assertEquals(1, explainedVariance.size());

        Vector explainedVarianceRatio = (Vector) params.get("explained_variance_ratio");
        assertEquals(1, explainedVarianceRatio.size());

        Vector singularValues = (Vector) params.get("singular_values");
        assertEquals(1, singularValues.size());

        Vector mean = (Vector) params.get("mean");
        assertEquals(2, mean.size());
    }

    @Test
    void testComponentsAlignWithPrincipalDirections() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0},
            {2.0, 2.0},
            {3.0, 3.0},
            {4.0, 4.0},
            {5.0, 5.0}
        });

        PCA pca = new PCA(1);
        pca.fit(X, null);

        Map<String, Object> params = pca.getParameters();
        Matrix components = (Matrix) params.get("components");
        Vector comp = components.row(0);

        assertEquals(2, comp.size());
        double ratio = comp.get(0) / comp.get(1);
        assertEquals(1.0, ratio, 1e-10);

        Vector mean = (Vector) params.get("mean");
        assertEquals(3.0, mean.get(0), 1e-10);
        assertEquals(3.0, mean.get(1), 1e-10);

        Vector explainedVarianceRatio = (Vector) params.get("explained_variance_ratio");
        assertEquals(1.0, explainedVarianceRatio.get(0), 1e-10);
    }

    @Test
    void testTransformFailsWhenNotFitted() {
        PCA pca = new PCA(2);
        Matrix X = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        assertThrows(IllegalStateException.class, () -> pca.transform(X));
    }

    @Test
    void testInverseTransformFailsWhenNotFitted() {
        PCA pca = new PCA(2);
        Matrix X = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        assertThrows(IllegalStateException.class, () -> pca.inverseTransform(X));
    }

    @Test
    void testTransformFailsOnWrongFeatureCount() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        });

        PCA pca = new PCA(2);
        pca.fit(X, null);

        Matrix wrongX = new Matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0}
        });

        assertThrows(IllegalArgumentException.class, () -> pca.transform(wrongX));
    }

    @Test
    void testSingleComponentVariance() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 4.0},
            {2.0, 5.0},
            {3.0, 6.0}
        });

        PCA pca = new PCA(2);
        pca.fit(X, null);

        Map<String, Object> params = pca.getParameters();
        Vector ev = (Vector) params.get("explained_variance");
        Vector evr = (Vector) params.get("explained_variance_ratio");

        assertEquals(2, ev.size());
        assertEquals(2, evr.size());

        double sum = evr.sum();
        assertEquals(1.0, sum, 1e-10);

        assertTrue(evr.get(0) >= evr.get(1));
    }
}
