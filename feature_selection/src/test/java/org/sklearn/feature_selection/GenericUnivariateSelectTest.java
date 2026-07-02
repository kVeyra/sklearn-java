package org.sklearn.feature_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class GenericUnivariateSelectTest {

    @Test
    void testKBest() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0}, {2, 0, 0}, {3, 1, 1}, {4, 1, 1}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        GenericUnivariateSelect sel = new GenericUnivariateSelect("k_best", 2, "f_classif");
        sel.fit(X, y);
        Matrix Xr = sel.transform(X);
        assertEquals(2, Xr.cols());
    }

    @Test
    void testPercentile() {
        Matrix X = new Matrix(new double[][]{
            {1, 0, 0}, {2, 0, 0}, {3, 1, 0}, {4, 1, 0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        GenericUnivariateSelect sel = new GenericUnivariateSelect("percentile", 50, "f_classif");
        sel.fit(X, y);
        Matrix Xr = sel.fitTransform(X, y);
        assertEquals(4, Xr.rows());
    }
}
