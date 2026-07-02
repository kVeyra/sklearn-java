package org.sklearn.model_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class GridSearchCVTest {

    public static class DummyClassifier implements Predictor<Matrix, Vector, Vector> {
        public double alpha = 1.0;
        public int maxIter = 100;

        @Override
        public DummyClassifier fit(Matrix X, Vector y) {
            return this;
        }

        @Override
        public Map<String, Object> getParameters() {
            Map<String, Object> p = new LinkedHashMap<>();
            p.put("alpha", alpha);
            p.put("maxIter", maxIter);
            return p;
        }

        @Override
        public Vector predict(Matrix X) {
            return new Vector(X.rows());
        }

        @Override
        public double score(Matrix X, Vector y) {
            double match = 0;
            for (int i = 0; i < y.size(); i++) {
                if (Math.abs(y.get(i)) < 0.5) {
                    match++;
                }
            }
            return match / y.size();
        }
    }

    @Test
    void testGridSearchFindsBestParams() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        Map<String, double[]> paramGrid = new LinkedHashMap<>();
        paramGrid.put("alpha", new double[]{0.5, 1.0, 2.0});
        paramGrid.put("maxIter", new double[]{50, 100});

        GridSearchCV gs = new GridSearchCV(new DummyClassifier(), paramGrid, 3, false, false);
        gs.fit(X, y);

        assertNotNull(gs.getBestParams());
        assertTrue(gs.getBestScore() > 0);
        assertNotNull(gs.getBestEstimator());
    }

    @Test
    void testGridSearchAllCombinations() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {3, 4}, {5, 6}, {7, 8}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        Map<String, double[]> paramGrid = new LinkedHashMap<>();
        paramGrid.put("alpha", new double[]{0.1, 1.0});
        paramGrid.put("maxIter", new double[]{50, 100, 200});

        GridSearchCV gs = new GridSearchCV(new DummyClassifier(), paramGrid, 2, false, false);
        gs.fit(X, y);

        // 2 * 3 = 6 combinations
        assertNotNull(gs.getCvResults());
        assertEquals(6, gs.getCvResults().size());
    }

    @Test
    void testGridSearchRefit() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {3, 4}, {5, 6}, {7, 8}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        Map<String, double[]> paramGrid = new LinkedHashMap<>();
        paramGrid.put("alpha", new double[]{1.0});

        GridSearchCV gs = new GridSearchCV(new DummyClassifier(), paramGrid, 2, false, true);
        gs.fit(X, y);

        Vector preds = gs.predict(X);
        assertNotNull(preds);
        assertEquals(4, preds.size());
    }

    @Test
    void testGridSearchScore() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {3, 4}, {5, 6}, {7, 8}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        Map<String, double[]> paramGrid = new LinkedHashMap<>();
        paramGrid.put("alpha", new double[]{1.0});

        GridSearchCV gs = new GridSearchCV(new DummyClassifier(), paramGrid, 2, false, true);
        gs.fit(X, y);

        double score = gs.score(X, y);
        assertTrue(score >= 0);
    }

    @Test
    void testPredictBeforeFitThrows() {
        Map<String, double[]> paramGrid = new LinkedHashMap<>();
        paramGrid.put("alpha", new double[]{1.0});

        GridSearchCV gs = new GridSearchCV(new DummyClassifier(), paramGrid, 2, false, false);
        assertThrows(IllegalStateException.class, () -> gs.predict(new Matrix(3, 2)));
    }
}
