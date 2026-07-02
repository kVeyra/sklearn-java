package org.sklearn.model_selection;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ParameterGridTest {

    @Test
    void testGenerate() {
        Map<String, Object[]> grid = Map.of(
            "C", new Object[]{0.1, 1.0, 10.0},
            "gamma", new Object[]{0.01, 0.1}
        );
        List<Map<String, Object>> combos = ParameterGrid.generate(grid);
        assertEquals(6, combos.size());
        for (Map<String, Object> combo : combos) {
            assertTrue(combo.containsKey("C"));
            assertTrue(combo.containsKey("gamma"));
        }
    }

    @Test
    void testEmpty() {
        Map<String, Object[]> grid = Map.of();
        List<Map<String, Object>> combos = ParameterGrid.generate(grid);
        assertEquals(1, combos.size());
    }
}
