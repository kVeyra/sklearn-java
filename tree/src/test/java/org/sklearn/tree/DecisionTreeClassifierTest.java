package org.sklearn.tree;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class DecisionTreeClassifierTest {

    @Test
    void testSimpleClassification() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        DecisionTreeClassifier clf = new DecisionTreeClassifier(3, 2, 1);
        clf.fit(X, y);

        assertEquals(1.0, clf.score(X, y), 1e-10);
    }

    @Test
    void testTwoFeatures() {
        Matrix X = new Matrix(new double[][]{
            {1, 1}, {2, 2}, {3, 3}, {10, 10}, {11, 11}, {12, 12}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        DecisionTreeClassifier clf = new DecisionTreeClassifier(3, 2, 1);
        clf.fit(X, y);

        assertEquals(1.0, clf.score(X, y), 1e-10);
    }

    @Test
    void testPredictProba() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {10.0}, {11.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        DecisionTreeClassifier clf = new DecisionTreeClassifier(3, 2, 1);
        clf.fit(X, y);

        Vector probs = clf.predictProba(X);
        assertEquals(4, probs.size());
    }

    @Test
    void testMaxDepth() {
        Matrix X = new Matrix(new double[][]{
            {1, 1}, {2, 2}, {3, 3}, {10, 10}, {11, 11}, {12, 12}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        DecisionTreeClassifier shallow = new DecisionTreeClassifier(1, 2, 1);
        shallow.fit(X, y);

        DecisionTreeClassifier deep = new DecisionTreeClassifier(10, 2, 1);
        deep.fit(X, y);

        // Deeper tree should have at least perfect accuracy
        assertTrue(deep.score(X, y) >= shallow.score(X, y));
    }

    @Test
    void testEntropyCriterion() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        DecisionTreeClassifier clf = new DecisionTreeClassifier(3, 2, 1, "entropy");
        clf.fit(X, y);

        assertEquals(1.0, clf.score(X, y), 1e-10);
    }

    @Test
    void testInvalidCriterionThrows() {
        assertThrows(IllegalArgumentException.class,
            () -> new DecisionTreeClassifier(3, 2, 1, "invalid"));
    }

    @Test
    void testPredictBeforeFitThrows() {
        DecisionTreeClassifier clf = new DecisionTreeClassifier(3, 2, 1);
        assertThrows(IllegalStateException.class,
            () -> clf.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{0, 1});

        DecisionTreeClassifier clf = new DecisionTreeClassifier(3, 2, 1);
        clf.fit(X, y);

        Matrix Xtest = new Matrix(new double[][]{{1, 2, 3}});
        assertThrows(IllegalArgumentException.class, () -> clf.predict(Xtest));
    }

    @Test
    void testXORProblem() {
        Matrix X = new Matrix(new double[][]{
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        });
        Vector y = new Vector(new double[]{0, 1, 1, 0});

        DecisionTreeClassifier clf = new DecisionTreeClassifier(5, 2, 1);
        clf.fit(X, y);

        // XOR requires at least 2 splits
        double score = clf.score(X, y);
        assertTrue(score >= 0.75);
    }

    @Test
    void testGetParameters() {
        DecisionTreeClassifier clf = new DecisionTreeClassifier(5, 4, 2, "entropy");
        var params = clf.getParameters();
        assertEquals(5, params.get("max_depth"));
        assertEquals(4, params.get("min_samples_split"));
        assertEquals(2, params.get("min_samples_leaf"));
        assertEquals("entropy", params.get("criterion"));
    }
}
