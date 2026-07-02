package org.sklearn.dummy;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class DummyClassifierTest {

    @Test
    void testMostFrequent() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}});
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        DummyClassifier dc = new DummyClassifier("most_frequent", null);
        dc.fit(X, y);
        Vector preds = dc.predict(X);
        for (int i = 0; i < 4; i++) {
            assertTrue(preds.get(i) == 0 || preds.get(i) == 1);
        }
        assertTrue(dc.isFitted());
    }

    @Test
    void testPrior() {
        Matrix X = new Matrix(new double[][]{{1}, {2}});
        Vector y = new Vector(new double[]{1, 1});

        DummyClassifier dc = new DummyClassifier("prior", null);
        dc.fit(X, y);
        Vector preds = dc.predict(X);
        assertEquals(1, preds.get(0));
    }

    @Test
    void testConstant() {
        Matrix X = new Matrix(new double[][]{{1}, {2}});
        Vector y = new Vector(new double[]{0, 0});

        DummyClassifier dc = new DummyClassifier("constant", 1.0);
        dc.fit(X, y);
        Vector preds = dc.predict(X);
        assertEquals(1, preds.get(0));
        assertEquals(1, preds.get(1));
    }

    @Test
    void testStratified() {
        Matrix X = new Matrix(new double[][]{{1}, {2}});
        Vector y = new Vector(new double[]{0, 1});

        DummyClassifier dc = new DummyClassifier("stratified", null);
        dc.fit(X, y);
        Vector preds = dc.predict(X);
        assertEquals(2, preds.size());
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}});
        Vector y = new Vector(new double[]{0, 0, 0});

        DummyClassifier dc = new DummyClassifier("most_frequent", null);
        dc.fit(X, y);
        double score = dc.score(X, y);
        assertEquals(1.0, score, 1e-12);
    }

    @Test
    void testParameters() {
        DummyClassifier dc = new DummyClassifier();
        assertTrue(dc.getParameters().containsKey("strategy"));
    }
}
