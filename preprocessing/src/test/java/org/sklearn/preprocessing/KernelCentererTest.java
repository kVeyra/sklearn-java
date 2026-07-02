package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class KernelCentererTest {

    @Test
    void testBasicFitTransform() {
        Matrix K = new Matrix(new double[][]{{4.0, 2.0, 1.0}, {2.0, 5.0, 3.0}, {1.0, 3.0, 6.0}});
        KernelCenterer kc = new KernelCenterer();
        kc.fit(K, null);
        assertTrue(kc.isFitted());
        Matrix Kc = kc.transform(K);
        assertEquals(3, Kc.rows());
        assertEquals(3, Kc.cols());
    }

    @Test
    void testColumnMeansZero() {
        Matrix K = new Matrix(new double[][]{{4.0, 2.0, 1.0}, {2.0, 5.0, 3.0}, {1.0, 3.0, 6.0}});
        KernelCenterer kc = new KernelCenterer();
        kc.fit(K, null);
        Matrix Kc = kc.transform(K);

        double[] colMeans = new double[Kc.cols()];
        for (int j = 0; j < Kc.cols(); j++) {
            double sum = 0;
            for (int i = 0; i < Kc.rows(); i++) sum += Kc.get(i, j);
            colMeans[j] = sum / Kc.rows();
        }
        for (int j = 0; j < colMeans.length; j++) {
            assertEquals(0.0, colMeans[j], 1e-12);
        }
    }

    @Test
    void testTransformBeforeFitThrows() {
        KernelCenterer kc = new KernelCenterer();
        assertThrows(IllegalStateException.class, () -> kc.transform(new Matrix(2, 2)));
    }

    @Test
    void testInverseTransformThrows() {
        KernelCenterer kc = new KernelCenterer();
        kc.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}), null);
        assertThrows(UnsupportedOperationException.class,
            () -> kc.inverseTransform(new Matrix(2, 2)));
    }

    @Test
    void testGetParameters() {
        KernelCenterer kc = new KernelCenterer();
        kc.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}), null);
        Map<String, Object> params = kc.getParameters();
        assertNotNull(params.get("K_fit_rows_"));
        assertNotNull(params.get("K_fit_all_"));
    }
}
