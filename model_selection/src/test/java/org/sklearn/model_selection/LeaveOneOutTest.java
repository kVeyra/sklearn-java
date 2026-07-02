package org.sklearn.model_selection;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class LeaveOneOutTest {

    @Test
    void testSplit() {
        LeaveOneOut loo = new LeaveOneOut();
        var splits = loo.split(4);
        assertEquals(4, splits.size());
        for (var split : splits) {
            assertEquals(3, split[0].length);
            assertEquals(1, split[1].length);
        }
    }

    @Test
    void testFirstFold() {
        LeaveOneOut loo = new LeaveOneOut();
        var splits = loo.split(3);
        var first = splits.get(0);
        assertEquals(0, first[1][0]);
        assertEquals(1, first[0][0]);
        assertEquals(2, first[0][1]);
    }
}
