package org.sklearn.isotonic;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class IsotonicRegressionTest {

    @Test
    void testBasicFit() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}});
        Vector y = new Vector(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        IsotonicRegression ir = new IsotonicRegression(true);
        ir.fit(X, y);
        assertTrue(ir.isFitted());
        Vector preds = ir.predict(X);
        assertEquals(5, preds.size());
        for (int i = 0; i < preds.size(); i++) {
            assertEquals(y.get(i), preds.get(i), 1e-12);
        }
    }

    @Test
    void testMonotonicIncreasing() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}});
        Vector y = new Vector(new double[]{1.0, 3.0, 2.0, 5.0, 4.0});
        IsotonicRegression ir = new IsotonicRegression(false);
        ir.fit(X, y);
        Vector preds = ir.predict(X);
        for (int i = 1; i < preds.size(); i++) {
            assertTrue(preds.get(i) >= preds.get(i - 1) - 1e-12);
        }
    }

    @Test
    void testDecreasing() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}});
        Vector y = new Vector(new double[]{5.0, 3.0, 1.0});
        IsotonicRegression ir = new IsotonicRegression(true);
        ir.fit(X, y);
        Vector preds = ir.predict(X);
        for (int i = 1; i < preds.size(); i++) {
            assertTrue(preds.get(i) <= preds.get(i - 1) + 1e-12);
        }
    }

    @Test
    void testPredictBeforeFitThrows() {
        IsotonicRegression ir = new IsotonicRegression(false);
        assertThrows(IllegalStateException.class,
            () -> ir.predict(new Matrix(1, 1)));
    }

    @Test
    void testFeatureMismatchThrows() {
        IsotonicRegression ir = new IsotonicRegression(false);
        ir.fit(new Matrix(new double[][]{{1.0}, {2.0}}),
            new Vector(new double[]{1.0, 2.0}));
        assertThrows(IllegalArgumentException.class,
            () -> ir.predict(new Matrix(new double[][]{{1.0, 2.0}})));
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}});
        Vector y = new Vector(new double[]{1.0, 2.0, 3.0});
        IsotonicRegression ir = new IsotonicRegression(true);
        ir.fit(X, y);
        double score = ir.score(X, y);
        assertEquals(1.0, score, 1e-12);
    }

    @Test
    void testGetParameters() {
        IsotonicRegression ir = new IsotonicRegression(true);
        assertEquals(true, ir.getParameters().get("increasing"));
    }

    @Test
    void testInterpolation() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}, {4.0}});
        Vector y = new Vector(new double[]{1.0, 2.0, 2.5, 4.0});
        IsotonicRegression ir = new IsotonicRegression(false);
        ir.fit(X, y);
        Matrix Xtest = new Matrix(new double[][]{{2.5}});
        Vector pred = ir.predict(Xtest);
        assertTrue(pred.get(0) >= 2.0);
        assertTrue(pred.get(0) <= 2.5);
    }
}
