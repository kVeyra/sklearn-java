package org.sklearn.model_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Vector;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class StratifiedKFoldTest {

    @Test
    void testBasicSplit() {
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});
        StratifiedKFold skf = new StratifiedKFold(3, false, 42);
        List<int[][]> folds = skf.split(y);
        assertEquals(3, folds.size());
        for (int[][] fold : folds) {
            Set<Integer> test = new HashSet<>();
            for (int i : fold[1]) {
                test.add(i);
            }
            assertEquals(2, test.size());
        }
    }

    @Test
    void testClassProportions() {
        Vector y = new Vector(new double[]{
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1
        });
        StratifiedKFold skf = new StratifiedKFold(3, false, 42);
        List<int[][]> folds = skf.split(y);
        for (int[][] fold : folds) {
            int count0 = 0;
            int count1 = 0;
            for (int i : fold[1]) {
                if (y.get(i) == 0) {
                    count0++;
                } else {
                    count1++;
                }
            }
            assertEquals(2, count0);
            assertEquals(2, count1);
        }
    }

    @Test
    void testNoOverlap() {
        Vector y = new Vector(new double[]{0, 0, 1, 1});
        StratifiedKFold skf = new StratifiedKFold(2, false, 42);
        List<int[][]> folds = skf.split(y);
        for (int[][] fold : folds) {
            Set<Integer> train = new HashSet<>();
            Set<Integer> test = new HashSet<>();
            for (int i : fold[0]) {
                train.add(i);
            }
            for (int i : fold[1]) {
                test.add(i);
            }
            train.retainAll(test);
            assertTrue(train.isEmpty());
        }
    }

    @Test
    void testShuffle() {
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});
        StratifiedKFold skf = new StratifiedKFold(3, true, 123);
        List<int[][]> folds = skf.split(y);
        assertEquals(3, folds.size());
    }

    @Test
    void testInvalidNSplitsThrows() {
        assertThrows(IllegalArgumentException.class, () -> new StratifiedKFold(1, false, 0));
    }
}
