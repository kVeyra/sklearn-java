package org.sklearn.model_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.core.Predictor;
import org.sklearn.linear_model.Ridge;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class RandomizedSearchCVTest {

    @Test
    void testRandomizedSearch() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}});
        Vector y = new Vector(new double[]{2, 3, 4, 5, 6});

        Map<String, Object[]> paramDist = new LinkedHashMap<>();
        paramDist.put("alpha", new Object[]{0.1, 1.0, 10.0});

        RandomizedSearchCV rscv = new RandomizedSearchCV(
            new Ridge(), paramDist, 3, 2, false, true, 42);
        rscv.fit(X, y);

        assertNotNull(rscv.getBestParams());
        assertTrue(rscv.getBestScore() > -Double.MAX_VALUE);
        assertNotNull(rscv.getBestEstimator());

        Vector preds = rscv.predict(X);
        assertEquals(5, preds.size());
    }
}
