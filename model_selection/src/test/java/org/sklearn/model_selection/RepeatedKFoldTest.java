package org.sklearn.model_selection;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class RepeatedKFoldTest {

    @Test
    void testRepeat() {
        RepeatedKFold rkf = new RepeatedKFold(3, 2, 42);
        var splits = rkf.split(9);
        assertEquals(6, splits.size());
    }

    @Test
    void testNSplits() {
        RepeatedKFold rkf = new RepeatedKFold(5, 10, 42);
        assertEquals(50, rkf.getNSplits());
    }

    @Test
    void testFoldSize() {
        RepeatedKFold rkf = new RepeatedKFold(3, 1, 0);
        var splits = rkf.split(6);
        for (var split : splits) {
            assertEquals(4, split[0].length);
            assertEquals(2, split[1].length);
        }
    }
}
