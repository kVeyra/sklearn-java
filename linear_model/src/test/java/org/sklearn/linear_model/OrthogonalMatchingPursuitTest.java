package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class OrthogonalMatchingPursuitTest {

    @Test
    void testFitPredict() {
        Matrix X = new Matrix(new double[][]{
            {1, 2, 3},
            {2, 3, 4},
            {3, 4, 5},
            {4, 5, 6}
        });
        Vector y = new Vector(new double[]{6, 9, 12, 15});

        OrthogonalMatchingPursuit omp = new OrthogonalMatchingPursuit(2);
        omp.fit(X, y);
        Vector preds = omp.predict(X);

        assertEquals(4, preds.size());
        assertNotNull(omp.getCoef());
    }

    @Test
    void testActiveIndices() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
        });
        Vector y = new Vector(new double[]{1, 2, 3});

        OrthogonalMatchingPursuit omp = new OrthogonalMatchingPursuit(2);
        omp.fit(X, y);
        assertTrue(omp.getActiveIndices().length > 0);
        double score = omp.score(X, y);
        assertFalse(Double.isNaN(score));
    }

    @Test
    void testThrowsOnInvalid() {
        assertThrows(IllegalArgumentException.class,
            () -> new OrthogonalMatchingPursuit(0));
    }
}
