package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class AdaBoostClassifierTest {

    @Test
    void testSimpleClassification() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        AdaBoostClassifier ada = new AdaBoostClassifier(20, 1, 42);
        ada.fit(X, y);

        assertEquals(1.0, ada.score(X, y), 0.01);
    }

    @Test
    void testPredictProba() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {10.0}, {11.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        AdaBoostClassifier ada = new AdaBoostClassifier(20, 1, 42);
        ada.fit(X, y);

        Vector probs = ada.predictProba(X);
        assertEquals(4, probs.size());
        for (int i = 0; i < probs.size(); i++) {
            assertTrue(probs.get(i) >= 0.0 && probs.get(i) <= 1.0);
        }
    }

    @Test
    void testPredictBeforeFitThrows() {
        AdaBoostClassifier ada = new AdaBoostClassifier(10, 1, 42);
        assertThrows(IllegalStateException.class,
            () -> ada.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{0, 1});

        AdaBoostClassifier ada = new AdaBoostClassifier(10, 1, 42);
        ada.fit(X, y);

        Matrix Xtest = new Matrix(new double[][]{{1, 2, 3}});
        assertThrows(IllegalArgumentException.class, () -> ada.predict(Xtest));
    }

    @Test
    void testGetParameters() {
        AdaBoostClassifier ada = new AdaBoostClassifier(30, 3, 0.5, 42);
        var params = ada.getParameters();
        assertEquals(30, params.get("n_estimators"));
        assertEquals(3, params.get("max_depth"));
        assertEquals(0.5, params.get("learning_rate"));
    }

    @Test
    void testLearningRateAffectsPrediction() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        AdaBoostClassifier ada1 = new AdaBoostClassifier(20, 1, 0.1, 42);
        ada1.fit(X, y);

        AdaBoostClassifier ada2 = new AdaBoostClassifier(20, 1, 1.0, 42);
        ada2.fit(X, y);

        assertEquals(1.0, ada2.score(X, y), 0.01);
    }

    @Test
    void testIsFitted() {
        AdaBoostClassifier ada = new AdaBoostClassifier(10, 1, 42);
        assertFalse(ada.isFitted());
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {10.0}, {11.0}});
        Vector y = new Vector(new double[]{0, 0, 1, 1});
        ada.fit(X, y);
        assertTrue(ada.isFitted());
    }
}
