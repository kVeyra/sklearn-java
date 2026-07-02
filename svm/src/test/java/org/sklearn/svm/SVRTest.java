package org.sklearn.svm;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class SVRTest {

    @Test
    void testLinearRegression() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}
        });
        Vector y = new Vector(new double[]{2.0, 4.0, 6.0});

        SVR svr = new SVR();
        svr.setKernel("linear").setC(10.0).setEpsilon(0.1).setTol(1e-3).setMaxIter(5000);
        svr.fit(X, y);

        double score = svr.score(X, y);
        assertTrue(score > 0.5);
    }

    @Test
    void testPredict() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}
        });
        Vector y = new Vector(new double[]{2.0, 4.0, 6.0});

        SVR svr = new SVR();
        svr.setKernel("linear").setC(100.0).setEpsilon(0.01).setMaxIter(5000);
        svr.fit(X, y);

        Vector preds = svr.predict(new Matrix(new double[][]{{4.0}}));
        assertTrue(preds.get(0) > 0);
    }

    @Test
    void testPredictBeforeFitThrows() {
        SVR svr = new SVR();
        assertThrows(IllegalStateException.class,
            () -> svr.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{1, 2});

        SVR svr = new SVR();
        svr.setKernel("linear").fit(X, y);

        assertThrows(IllegalArgumentException.class,
            () -> svr.predict(new Matrix(new double[][]{{1, 2, 3}})));
    }

    @Test
    void testGetParameters() {
        SVR svr = new SVR();
        svr.setC(2.0).setEpsilon(0.2).setKernel("poly").setDegree(3);

        var params = svr.getParameters();
        assertEquals(2.0, params.get("C"));
        assertEquals(0.2, params.get("epsilon"));
        assertEquals("poly", params.get("kernel"));
        assertEquals(3, params.get("degree"));
    }
}
