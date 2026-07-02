package org.sklearn.semi_supervised;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class LabelPropagationTest {

    @Test
    void testBasicFit() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {5.0, 5.0}, {6.0, 6.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});
        LabelPropagation lp = new LabelPropagation();
        lp.fit(X, y);
        assertTrue(lp.isFitted());
        Vector preds = lp.predict(X);
        assertEquals(4, preds.size());
    }

    @Test
    void testSemiSupervised() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {5.0, 5.0}, {6.0, 6.0}
        });
        Vector y = new Vector(new double[]{0, -1, 1, -1});
        LabelPropagation lp = new LabelPropagation(200, 1e-4, 10.0);
        lp.fit(X, y);
        Vector preds = lp.predict(X);
        assertEquals(4, preds.size());
        for (int i = 0; i < preds.size(); i++) {
            assertTrue(preds.get(i) >= 0);
        }
    }

    @Test
    void testPredictNewData() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {5.0, 5.0}, {6.0, 6.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});
        LabelPropagation lp = new LabelPropagation();
        lp.fit(X, y);
        Matrix Xnew = new Matrix(new double[][]{{1.5, 1.5}, {5.5, 5.5}});
        Vector preds = lp.predict(Xnew);
        assertEquals(2, preds.size());
    }

    @Test
    void testPredictBeforeFitThrows() {
        LabelPropagation lp = new LabelPropagation();
        assertThrows(IllegalStateException.class,
            () -> lp.predict(new Matrix(1, 2)));
    }

    @Test
    void testFeatureMismatchThrows() {
        LabelPropagation lp = new LabelPropagation();
        lp.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}),
            new Vector(new double[]{0, 1}));
        assertThrows(IllegalArgumentException.class,
            () -> lp.predict(new Matrix(new double[][]{{1.0}, {2.0}})));
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {5.0, 5.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1});
        LabelPropagation lp = new LabelPropagation();
        lp.fit(X, y);
        double score = lp.score(X, y);
        assertTrue(score > 0.5);
    }

    @Test
    void testGetLabelDistributions() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}
        });
        Vector y = new Vector(new double[]{0, 0});
        LabelPropagation lp = new LabelPropagation();
        lp.fit(X, y);
        assertNotNull(lp.getLabelDistributions());
    }
}
