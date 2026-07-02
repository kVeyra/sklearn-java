package org.sklearn.model_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class CrossValidationTest {

    static class DummyClassifier implements Predictor<Matrix, Vector, Vector> {
        private Vector classes;

        @Override
        public DummyClassifier fit(Matrix X, Vector y) {
            classes = new Vector(new double[]{0, 1});
            return this;
        }

        @Override
        public Map<String, Object> getParameters() {
            return new LinkedHashMap<>();
        }

        @Override
        public Vector predict(Matrix X) {
            return new Vector(X.rows());
        }

        @Override
        public double score(Matrix X, Vector y) {
            return 1.0;
        }

        public Vector getClasses() {
            return classes;
        }
    }

    @Test
    void testCrossValScoreKFold() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        KFold cv = new KFold(3, false, 42);
        double[] scores = CrossValidation.crossValScore(new DummyClassifier(), X, y, cv);
        assertEquals(3, scores.length);
        for (double s : scores) {
            assertEquals(1.0, s, 1e-10);
        }
    }

    @Test
    void testCrossValScoreStratifiedKFold() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {3, 4}, {4, 5}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        StratifiedKFold cv = new StratifiedKFold(2, false, 42);
        double[] scores = CrossValidation.crossValScore(new DummyClassifier(), X, y, cv);
        assertEquals(2, scores.length);
    }

    @Test
    void testCrossValScoreReturnsScores() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}, {7, 8}});
        Vector y = new Vector(new double[]{0, 1, 0, 1});

        KFold cv = new KFold(2, false, 42);
        double[] scores = CrossValidation.crossValScore(new DummyClassifier(), X, y, cv);
        assertNotNull(scores);
        assertEquals(2, scores.length);
    }
}
