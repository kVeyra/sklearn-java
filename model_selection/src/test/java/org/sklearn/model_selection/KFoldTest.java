package org.sklearn.model_selection;

import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class KFoldTest {

    @Test
    void testBasicSplit() {
        KFold kf = new KFold(3, false, 42);
        List<int[][]> folds = kf.split(9);
        assertEquals(3, folds.size());
        for (int[][] fold : folds) {
            assertEquals(6, fold[0].length);
            assertEquals(3, fold[1].length);
        }
    }

    @Test
    void testNoOverlap() {
        KFold kf = new KFold(3, false, 42);
        List<int[][]> folds = kf.split(6);
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
    void testAllIndicesCovered() {
        KFold kf = new KFold(3, false, 42);
        List<int[][]> folds = kf.split(9);
        for (int[][] fold : folds) {
            Set<Integer> all = new HashSet<>();
            for (int i : fold[0]) {
                all.add(i);
            }
            for (int i : fold[1]) {
                all.add(i);
            }
            for (int i = 0; i < 9; i++) {
                assertTrue(all.contains(i));
            }
        }
    }

    @Test
    void testShuffle() {
        KFold kf = new KFold(3, true, 42);
        List<int[][]> folds = kf.split(10);
        assertEquals(3, folds.size());
        assertEquals(3, folds.get(0)[1].length);
    }

    @Test
    void testFewerSamplesThanSplitsThrows() {
        KFold kf = new KFold(5, false, 42);
        assertThrows(IllegalArgumentException.class, () -> kf.split(3));
    }

    @Test
    void testInvalidNSplitsThrows() {
        assertThrows(IllegalArgumentException.class, () -> new KFold(1, false, 0));
    }

    @Test
    void testNSplits() {
        KFold kf = new KFold(5, false, 42);
        assertEquals(5, kf.getNSplits());
    }
}
