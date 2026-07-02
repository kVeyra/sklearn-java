package org.sklearn.model_selection;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ShuffleSplitTest {

    @Test
    void testSplit() {
        ShuffleSplit ss = new ShuffleSplit(3, 0.2, 42);
        List<int[][]> splits = ss.split(100);
        assertEquals(3, splits.size());
        for (int[][] fold : splits) {
            assertEquals(2, fold.length);
            assertEquals(80, fold[0].length);
            assertEquals(20, fold[1].length);
        }
    }

    @Test
    void testDeterministic() {
        ShuffleSplit ss1 = new ShuffleSplit(2, 0.3, 123);
        ShuffleSplit ss2 = new ShuffleSplit(2, 0.3, 123);
        List<int[][]> s1 = ss1.split(50);
        List<int[][]> s2 = ss2.split(50);
        for (int f = 0; f < 2; f++) {
            assertArrayEquals(s1.get(f)[0], s2.get(f)[0]);
            assertArrayEquals(s1.get(f)[1], s2.get(f)[1]);
        }
    }

    @Test
    void testMinSamples() {
        ShuffleSplit ss = new ShuffleSplit(1, 0.5, 0);
        List<int[][]> splits = ss.split(2);
        assertEquals(1, splits.size());
        assertEquals(1, splits.get(0)[0].length);
        assertEquals(1, splits.get(0)[1].length);
    }
}
