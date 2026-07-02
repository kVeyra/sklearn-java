package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class RandomForestClassifierTest {

    @Test
    void testSimpleClassification() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        RandomForestClassifier rf = new RandomForestClassifier(10, 3, 2, 1);
        rf.fit(X, y);

        assertEquals(1.0, rf.score(X, y), 0.01);
    }

    @Test
    void testPredictProba() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {10.0}, {11.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        RandomForestClassifier rf = new RandomForestClassifier(10, 3, 2, 1);
        rf.fit(X, y);

        Vector probs = rf.predictProba(X);
        assertEquals(4, probs.size());
        for (int i = 0; i < probs.size(); i++) {
            assertTrue(probs.get(i) >= 0.0 && probs.get(i) <= 1.0);
        }
    }

    @Test
    void testPredictBeforeFitThrows() {
        RandomForestClassifier rf = new RandomForestClassifier(10, 3, 2, 1);
        assertThrows(IllegalStateException.class,
            () -> rf.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{0, 1});

        RandomForestClassifier rf = new RandomForestClassifier(10, 3, 2, 1);
        rf.fit(X, y);

        Matrix Xtest = new Matrix(new double[][]{{1, 2, 3}});
        assertThrows(IllegalArgumentException.class, () -> rf.predict(Xtest));
    }

    @Test
    void testGetParameters() {
        RandomForestClassifier rf = new RandomForestClassifier(50, 5, 4, 2);
        var params = rf.getParameters();
        assertEquals(50, params.get("n_estimators"));
        assertEquals(5, params.get("max_depth"));
        assertEquals(4, params.get("min_samples_split"));
        assertEquals(2, params.get("min_samples_leaf"));
    }
}
