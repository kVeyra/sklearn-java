package org.sklearn.model_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Vector;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class StratifiedShuffleSplitTest {

    @Test
    void testSplit() {
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});
        StratifiedShuffleSplit sss = new StratifiedShuffleSplit(2, 0.3, 42);
        List<int[][]> splits = sss.split(y);
        assertEquals(2, splits.size());
        for (int[][] fold : splits) {
            assertTrue(fold[0].length > 0);
            assertTrue(fold[1].length > 0);
        }
    }

    @Test
    void testDeterministic() {
        Vector y = new Vector(new double[]{0, 0, 0, 0, 1, 1, 1, 1});
        StratifiedShuffleSplit sss1 = new StratifiedShuffleSplit(2, 0.25, 99);
        StratifiedShuffleSplit sss2 = new StratifiedShuffleSplit(2, 0.25, 99);
        List<int[][]> s1 = sss1.split(y);
        List<int[][]> s2 = sss2.split(y);
        for (int f = 0; f < 2; f++) {
            assertArrayEquals(s1.get(f)[0], s2.get(f)[0]);
            assertArrayEquals(s1.get(f)[1], s2.get(f)[1]);
        }
    }
}
