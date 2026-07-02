package org.sklearn.naive_bayes;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class BernoulliNBTest {

    @Test
    void testBasic() {
        Matrix X = new Matrix(new double[][]{{1, 0, 1}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}});
        Vector y = new Vector(new double[]{0, 1, 0, 1});
        BernoulliNB nb = new BernoulliNB();
        nb.fit(X, y);
        assertTrue(nb.score(X, y) > 0);
    }

    @Test
    void testBinarization() {
        Matrix X = new Matrix(new double[][]{{0.5, 1.5}, {-0.5, 0.2}, {0.1, -0.3}});
        Vector y = new Vector(new double[]{0, 0, 1});
        BernoulliNB nb = new BernoulliNB(1.0, true, null, true, 0.0);
        nb.fit(X, y);
        assertNotNull(nb.getFeatureLogProb());
    }

    @Test
    void testPredictProbas() {
        Matrix X = new Matrix(new double[][]{{1, 0}, {0, 1}, {1, 1}, {0, 0}});
        Vector y = new Vector(new double[]{0, 1, 0, 1});
        BernoulliNB nb = new BernoulliNB();
        nb.fit(X, y);
        double[] probas = nb.predictProbas(X);
        assertEquals(8, probas.length);
    }

    @Test
    void testFitPriorFalse() {
        Matrix X = new Matrix(new double[][]{{1, 0}, {0, 1}});
        Vector y = new Vector(new double[]{0, 1});
        BernoulliNB nb = new BernoulliNB(1.0, false, null, false, 0.0);
        nb.fit(X, y);
        assertEquals(2, nb.getClasses().length);
    }

    @Test
    void testCustomPriors() {
        Matrix X = new Matrix(new double[][]{{1, 0}, {0, 1}, {1, 1}});
        Vector y = new Vector(new double[]{0, 1, 0});
        BernoulliNB nb = new BernoulliNB(1.0, true, new double[]{0.7, 0.3}, false, 0.0);
        nb.fit(X, y);
        assertArrayEquals(new int[]{0, 1}, nb.getClasses());
    }

    @Test
    void testPredictBeforeFitThrows() {
        BernoulliNB nb = new BernoulliNB();
        assertThrows(IllegalStateException.class, () -> nb.predict(new Matrix(1, 1)));
    }
}
