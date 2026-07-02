package org.sklearn.feature_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class SelectFweTest {

    @Test
    void testBasicSelection() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 10.0, 100.0},
            {1.1, 20.0, 200.0},
            {5.0, 30.0, 300.0},
            {5.1, 40.0, 400.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});
        SelectFwe selector = new SelectFwe(0.5, "f_classif");
        selector.fit(X, y);
        assertTrue(selector.isFitted());
        Matrix Xt = selector.transform(X);
        assertTrue(Xt.cols() >= 1);
    }

    @Test
    void testFRegression() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 5.0}, {2.0, 6.0}, {3.0, 7.0}, {4.0, 8.0}, {5.0, 9.0}
        });
        Vector y = new Vector(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        SelectFwe selector = new SelectFwe(0.5, "f_regression");
        selector.fit(X, y);
        Matrix Xt = selector.transform(X);
        assertTrue(Xt.cols() >= 1);
    }

    @Test
    void testHighAlphaSelectsAll() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}
        });
        Vector y = new Vector(new double[]{0, 1});
        SelectFwe selector = new SelectFwe(1.0, "f_classif");
        selector.fit(X, y);
        assertEquals(3, selector.getSupport().length);
    }

    @Test
    void testTransformBeforeFitThrows() {
        SelectFwe selector = new SelectFwe();
        assertThrows(IllegalStateException.class,
            () -> selector.transform(new Matrix(2, 3)));
    }

    @Test
    void testInverseTransformThrows() {
        SelectFwe selector = new SelectFwe();
        selector.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}),
            new Vector(new double[]{0, 1}));
        assertThrows(UnsupportedOperationException.class,
            () -> selector.inverseTransform(new Matrix(2, 1)));
    }

    @Test
    void testFitTransform() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 10.0}, {2.0, 11.0}, {5.0, 12.0}, {6.0, 13.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});
        SelectFwe selector = new SelectFwe(0.5, "f_classif");
        Matrix Xt = selector.fitTransform(X, y);
        assertTrue(Xt.cols() >= 1);
    }

    @Test
    void testGetParameters() {
        SelectFwe selector = new SelectFwe(0.01, "f_regression");
        assertEquals(0.01, selector.getParameters().get("alpha"));
        assertEquals("f_regression", selector.getParameters().get("score_func"));
    }
}
