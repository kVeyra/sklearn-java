package org.sklearn.naive_bayes;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class MultinomialNBTest {

    @Test
    void testBasic() {
        Matrix X = new Matrix(new double[][]{{2, 1, 0}, {0, 1, 2}, {1, 1, 1}, {2, 0, 2}});
        Vector y = new Vector(new double[]{0, 1, 0, 1});
        MultinomialNB nb = new MultinomialNB();
        nb.fit(X, y);
        assertTrue(nb.score(X, y) > 0);
    }

    @Test
    void testProbPredict() {
        Matrix X = new Matrix(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}});
        Vector y = new Vector(new double[]{0, 1, 0, 1});
        MultinomialNB nb = new MultinomialNB(0.5, false, null);
        nb.fit(X, y);
        double[] probas = nb.predictProbas(X);
        assertEquals(8, probas.length);
        for (double p : probas) {
            assertTrue(p >= 0 && p <= 1);
        }
    }

    @Test
    void testAlphaSmoothing() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}, {5}});
        Vector y = new Vector(new double[]{0, 0, 1, 1, 1});
        MultinomialNB nb = new MultinomialNB(1.0, true, null);
        nb.fit(X, y);
        assertNotNull(nb.getFeatureLogProb());
    }

    @Test
    void testNegativeValuesThrow() {
        Matrix X = new Matrix(new double[][]{{1, -1}, {2, 3}});
        Vector y = new Vector(new double[]{0, 1});
        MultinomialNB nb = new MultinomialNB();
        assertThrows(IllegalArgumentException.class, () -> nb.fit(X, y));
    }

    @Test
    void testPredictBeforeFitThrows() {
        MultinomialNB nb = new MultinomialNB();
        assertThrows(IllegalStateException.class, () -> nb.predict(new Matrix(1, 1)));
    }
}
