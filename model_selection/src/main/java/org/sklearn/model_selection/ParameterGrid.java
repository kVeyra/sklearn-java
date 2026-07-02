package org.sklearn.model_selection;

import java.util.*;

/**
 * Grid of parameters for search.
 *
 * <p>Generates all combinations of a parameter grid as a list of
 * maps from parameter name to value.
 *
 * <p>Mirrors {@code sklearn.model_selection.ParameterGrid}.
 */
public final class ParameterGrid {

    private ParameterGrid() {
    }

    /**
     * Generate all combinations of the given parameter grid.
     *
     * @param paramGrid map of parameter names to arrays of values
     * @return list of parameter dictionaries representing each combination
     */
    public static List<Map<String, Object>> generate(Map<String, Object[]> paramGrid) {
        List<String> names = new ArrayList<>(paramGrid.keySet());
        List<Object[]> values = new ArrayList<>();
        for (String name : names) {
            values.add(paramGrid.get(name));
        }

        List<Map<String, Object>> result = new ArrayList<>();
        cartesian(names, values, 0, new LinkedHashMap<>(), result);
        return result;
    }

    private static void cartesian(List<String> names, List<Object[]> values,
                                   int depth, Map<String, Object> current,
                                   List<Map<String, Object>> result) {
        if (depth == names.size()) {
            result.add(new LinkedHashMap<>(current));
            return;
        }
        for (Object v : values.get(depth)) {
            current.put(names.get(depth), v);
            cartesian(names, values, depth + 1, current, result);
            current.remove(names.get(depth));
        }
    }
}
