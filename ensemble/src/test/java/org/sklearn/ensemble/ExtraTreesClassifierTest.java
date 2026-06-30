package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class ExtraTreesClassifierTest {

    @Test
    void testSimpleClassification() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        ExtraTreesClassifier et = new ExtraTreesClassifier(10, 3, 2, 1, 42);
        et.fit(X, y);

        assertEquals(1.0, et.score(X, y), 0.01);
    }

    @Test
    void testPredictProba() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {10.0}, {11.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        ExtraTreesClassifier et = new ExtraTreesClassifier(10, 3, 2, 1, 42);
        et.fit(X, y);

        Vector probs = et.predictProba(X);
        assertEquals(4, probs.size());
        for (int i = 0; i < probs.size(); i++) {
            assertTrue(probs.get(i) >= 0.0 && probs.get(i) <= 1.0);
        }
    }

    @Test
    void testPredictBeforeFitThrows() {
        ExtraTreesClassifier et = new ExtraTreesClassifier(10, 3, 2, 1, 42);
        assertThrows(IllegalStateException.class,
            () -> et.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{0, 1});

        ExtraTreesClassifier et = new ExtraTreesClassifier(10, 3, 2, 1, 42);
        et.fit(X, y);

        Matrix Xtest = new Matrix(new double[][]{{1, 2, 3}});
        assertThrows(IllegalArgumentException.class, () -> et.predict(Xtest));
    }

    @Test
    void testGetParameters() {
        ExtraTreesClassifier et = new ExtraTreesClassifier(50, 5, 4, 2, 42);
        var params = et.getParameters();
        assertEquals(50, params.get("n_estimators"));
        assertEquals(5, params.get("max_depth"));
        assertEquals(4, params.get("min_samples_split"));
        assertEquals(2, params.get("min_samples_leaf"));
    }

    @Test
    void testIsFitted() {
        ExtraTreesClassifier et = new ExtraTreesClassifier(10, 3, 2, 1, 42);
        assertFalse(et.isFitted());
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {10.0}, {11.0}});
        Vector y = new Vector(new double[]{0, 0, 1, 1});
        et.fit(X, y);
        assertTrue(et.isFitted());
    }
}
