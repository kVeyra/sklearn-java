package org.sklearn.feature_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class SelectKBestTest {

    @Test
    void testSelectTopFeaturesClassif() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0},
            {2, 0, 0},
            {3, 1, 1},
            {4, 1, 1}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        SelectKBest selector = new SelectKBest(1, "f_classif");
        selector.fit(X, y);
        Matrix Xr = selector.transform(X);

        assertEquals(4, Xr.rows());
        assertEquals(1, Xr.cols());
        int[] idx = selector.getSelectedIndices();
        assertEquals(1, idx.length);
    }

    @Test
    void testSelectTopFeaturesRegression() {
        Matrix X = new Matrix(new double[][]{
            {1, 10},
            {2, 9},
            {3, 8},
            {4, 7}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        SelectKBest selector = new SelectKBest(1, "f_regression");
        selector.fit(X, y);
        Matrix Xr = selector.transform(X);

        assertEquals(4, Xr.rows());
        assertEquals(1, Xr.cols());
    }

    @Test
    void testChi2() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1},
            {1, 1, 0}
        });
        Vector y = new Vector(new double[]{0, 1, 2, 0});

        SelectKBest selector = new SelectKBest(2, "chi2");
        selector.fit(X, y);
        Matrix Xr = selector.transform(X);

        assertEquals(4, Xr.rows());
        assertEquals(2, Xr.cols());
    }

    @Test
    void testScoresAndPValues() {
        Matrix X = new Matrix(new double[][]{
            {1, 0}, {2, 0}, {3, 1}, {4, 1}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        SelectKBest selector = new SelectKBest(2, "f_classif");
        selector.fit(X, y);
        double[] scores = selector.getScores();
        double[] pvals = selector.getPValues();
        assertEquals(2, scores.length);
        assertEquals(2, pvals.length);
        assertTrue(scores[0] >= 0);
        assertTrue(pvals[0] >= 0 && pvals[0] <= 1);
    }

    @Test
    void testGetSupport() {
        Matrix X = new Matrix(new double[][]{
            {5, 1}, {6, 2}, {7, 3}, {8, 4}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        SelectKBest selector = new SelectKBest(1, "f_classif");
        selector.fit(X, y);
        int[] support = selector.getSupport();
        assertEquals(1, support.length);
    }

    @Test
    void testFitTransform() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0}, {2, 0, 0}, {3, 1, 1}, {4, 1, 1}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        SelectKBest selector = new SelectKBest(1, "f_classif");
        Matrix Xr = selector.fitTransform(X, y);
        assertEquals(4, Xr.rows());
        assertEquals(1, Xr.cols());
    }

    @Test
    void testInverseTransformThrows() {
        Matrix X = new Matrix(new double[][]{
            {1, 0}, {2, 0}, {3, 1}, {4, 1}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        SelectKBest selector = new SelectKBest(1, "f_classif");
        selector.fit(X, y);
        assertThrows(UnsupportedOperationException.class, () -> selector.inverseTransform(X));
    }
}
