package org.sklearn.svm;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class SVCTest {

    @Test
    void testLinearlySeparableBinary() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {2.0, 1.0},
            {5.0, 5.0}, {6.0, 6.0}, {5.0, 4.0}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        SVC svc = new SVC();
        svc.setKernel("linear").setC(1.0);
        svc.fit(X, y);

        assertEquals(1.0, svc.score(X, y), 0.01);
    }

    @Test
    void testPredictBinary() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 0.0}, {0.0, 1.0},
            {10.0, 0.0}, {0.0, 10.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        SVC svc = new SVC();
        svc.setKernel("linear").setC(10.0);
        svc.fit(X, y);

        Vector preds = svc.predict(new Matrix(new double[][]{
            {1.1, 0.0}, {10.1, 0.0}
        }));
        assertEquals(0.0, preds.get(0), 0.01);
        assertEquals(1.0, preds.get(1), 0.01);
    }

    @Test
    void testRbfKernel() {
        // Non-linear XOR-like data
        Matrix X = new Matrix(new double[][]{
            {0.0, 0.0}, {1.0, 1.0},
            {0.0, 1.0}, {1.0, 0.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        SVC svc = new SVC();
        svc.setKernel("rbf").setGamma("auto").setC(10.0);
        svc.fit(X, y);

        double score = svc.score(X, y);
        assertTrue(score >= 0.5);
    }

    @Test
    void testMultiClass() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0},
            {5.0, 5.0}, {6.0, 6.0},
            {9.0, 9.0}, {10.0, 10.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1, 2, 2});

        SVC svc = new SVC();
        svc.setKernel("linear").setC(10.0);
        svc.fit(X, y);

        double score = svc.score(X, y);
        assertTrue(score > 0.8);
    }

    @Test
    void testPredictBeforeFitThrows() {
        SVC svc = new SVC();
        assertThrows(IllegalStateException.class,
            () -> svc.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{0, 1});

        SVC svc = new SVC();
        svc.setKernel("linear").fit(X, y);

        assertThrows(IllegalArgumentException.class,
            () -> svc.predict(new Matrix(new double[][]{{1, 2, 3}})));
    }

    @Test
    void testPolynomialKernel() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {1.5, 1.0},
            {5.0, 5.0}, {5.5, 5.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        SVC svc = new SVC();
        svc.setKernel("poly").setDegree(2).setC(10.0);
        svc.fit(X, y);

        double score = svc.score(X, y);
        assertTrue(score > 0.5);
    }

    @Test
    void testGetParameters() {
        SVC svc = new SVC();
        svc.setC(2.0).setKernel("poly").setDegree(3).setGamma(0.5).setCoef0(1.0).setTol(1e-4);

        var params = svc.getParameters();
        assertEquals(2.0, params.get("C"));
        assertEquals("poly", params.get("kernel"));
        assertEquals(3, params.get("degree"));
        assertEquals(0.5, params.get("gamma"));
        assertEquals(1.0, params.get("coef0"));
        assertEquals(1e-4, params.get("tol"));
    }
}
