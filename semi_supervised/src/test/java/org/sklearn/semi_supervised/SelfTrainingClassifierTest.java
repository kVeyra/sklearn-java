package org.sklearn.semi_supervised;

import org.junit.jupiter.api.Test;
import org.sklearn.linear_model.LogisticRegression;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class SelfTrainingClassifierTest {

    @Test
    void testBasicFit() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {5.0, 5.0}, {6.0, 6.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});
        SelfTrainingClassifier st = new SelfTrainingClassifier(
            new LogisticRegression(), 0.8, 10);
        st.fit(X, y);
        assertTrue(st.isFitted());
        Vector preds = st.predict(X);
        assertEquals(4, preds.size());
    }

    @Test
    void testSemiSupervised() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0},
            {5.0, 5.0}, {6.0, 6.0}, {7.0, 7.0}
        });
        Vector y = new Vector(new double[]{0, -1, 0, 1, -1, 1});
        SelfTrainingClassifier st = new SelfTrainingClassifier(
            new LogisticRegression(), 0.5, 10);
        st.fit(X, y);
        Vector preds = st.predict(X);
        assertEquals(6, preds.size());
    }

    @Test
    void testPredictBeforeFitThrows() {
        SelfTrainingClassifier st = new SelfTrainingClassifier(
            new LogisticRegression(), 0.8, 10);
        assertThrows(IllegalStateException.class,
            () -> st.predict(new Matrix(1, 2)));
    }

    @Test
    void testFeatureMismatchThrows() {
        SelfTrainingClassifier st = new SelfTrainingClassifier(
            new LogisticRegression(), 0.8, 10);
        st.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}),
            new Vector(new double[]{0, 1}));
        assertThrows(IllegalArgumentException.class,
            () -> st.predict(new Matrix(new double[][]{{1.0}, {2.0}})));
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 2.0}, {5.0, 5.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1});
        SelfTrainingClassifier st = new SelfTrainingClassifier(
            new LogisticRegression(), 0.8, 10);
        st.fit(X, y);
        double score = st.score(X, y);
        assertTrue(score > 0.5);
    }

    @Test
    void testGetParameters() {
        SelfTrainingClassifier st = new SelfTrainingClassifier(
            new LogisticRegression(), 0.75, 20);
        st.fit(new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}}),
            new Vector(new double[]{0, 1}));
        assertEquals(0.75, st.getParameters().get("threshold"));
    }
}
