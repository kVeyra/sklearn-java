package org.sklearn.multiclass;

import org.junit.jupiter.api.Test;
import org.sklearn.linear_model.LogisticRegression;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class OneVsOneClassifierTest {

    @Test
    void testBasicClassification() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {5.0, 5.0}, {6.0, 6.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});
        OneVsOneClassifier ovo = new OneVsOneClassifier(new LogisticRegression());
        ovo.fit(X, y);
        assertTrue(ovo.isFitted());
        Vector preds = ovo.predict(X);
        assertEquals(y.get(0), preds.get(0), 0.0);
        assertEquals(y.get(2), preds.get(2), 0.0);
    }

    @Test
    void testThreeClasses() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {5.0, 5.0}, {6.0, 6.0}, {9.0, 9.0}, {10.0, 10.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1, 2, 2});
        OneVsOneClassifier ovo = new OneVsOneClassifier(new LogisticRegression());
        ovo.fit(X, y);
        Vector preds = ovo.predict(X);
        assertEquals(6, preds.size());
        int correct = 0;
        for (int i = 0; i < y.size(); i++) {
            if (y.get(i) == preds.get(i)) correct++;
        }
        assertTrue(correct >= 4);
    }

    @Test
    void testPredictBeforeFitThrows() {
        OneVsOneClassifier ovo = new OneVsOneClassifier(new LogisticRegression());
        assertThrows(IllegalStateException.class,
            () -> ovo.predict(new Matrix(1, 2)));
    }

    @Test
    void testFeatureMismatchThrows() {
        OneVsOneClassifier ovo = new OneVsOneClassifier(new LogisticRegression());
        ovo.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}),
            new Vector(new double[]{0, 1}));
        assertThrows(IllegalArgumentException.class,
            () -> ovo.predict(new Matrix(new double[][]{{1.0}, {2.0}})));
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {5.0, 5.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1});
        OneVsOneClassifier ovo = new OneVsOneClassifier(new LogisticRegression());
        ovo.fit(X, y);
        double score = ovo.score(X, y);
        assertTrue(score > 0.5);
    }

    @Test
    void testGetParameters() {
        OneVsOneClassifier ovo = new OneVsOneClassifier(new LogisticRegression());
        ovo.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}),
            new Vector(new double[]{0, 1}));
        Map<String, Object> params = ovo.getParameters();
        assertNotNull(params.get("classes"));
        assertNotNull(params.get("n_estimators"));
    }
}
