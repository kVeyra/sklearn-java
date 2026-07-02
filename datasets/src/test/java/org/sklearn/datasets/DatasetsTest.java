package org.sklearn.datasets;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DatasetsTest {

    @Test
    void testIrisDataset() {
        IrisDataset.Dataset ds = IrisDataset.load();
        assertEquals(150, ds.data.rows());
        assertEquals(4, ds.data.cols());
        assertEquals(150, ds.target.size());
    }

    @Test
    void testIrisTuple() {
        IrisDataset.Tuple t = IrisDataset.loadXY();
        assertEquals(150, t.x.rows());
        assertEquals(150, t.y.size());
    }

    @Test
    void testMakeClassification() {
        MakeClassification.Result r = MakeClassification.makeClassification(100, 5, 3, 42);
        assertEquals(100, r.x.rows());
        assertEquals(5, r.x.cols());
        assertEquals(100, r.y.size());
    }

    @Test
    void testMakeRegression() {
        MakeRegression.Result r = MakeRegression.makeRegression(50, 4, 0.1, 42);
        assertEquals(50, r.x.rows());
        assertEquals(4, r.x.cols());
        assertEquals(50, r.y.size());
    }

    @Test
    void testMakeBlobs() {
        MakeBlobs.Result r = MakeBlobs.makeBlobs(300, 2, 3, 1.0, 42);
        assertEquals(300, r.x.rows());
        assertEquals(2, r.x.cols());
        assertEquals(300, r.y.size());
    }

    @Test
    void testMakeMoons() {
        MakeMoons.Result r = MakeMoons.makeMoons(100, 0.1, 42);
        assertEquals(100, r.x.rows());
        assertEquals(2, r.x.cols());
    }

    @Test
    void testMakeCircles() {
        MakeCircles.Result r = MakeCircles.makeCircles(100, 0.05, 0.5, 42);
        assertEquals(100, r.x.rows());
        assertEquals(2, r.x.cols());
    }
}
