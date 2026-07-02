package org.sklearn.pipeline;

import java.util.*;

public final class PipelineUtils {

    private PipelineUtils() {}

    @SuppressWarnings("unchecked")
    public static Pipeline<?, ?, ?> makePipeline(Object... steps) {
        List<Object> stepList = new ArrayList<>();
        for (int i = 0; i < steps.length; i++) {
            if (steps[i] instanceof String && (i + 1) < steps.length) {
                stepList.add(steps[i + 1]);
                i++;
            } else {
                stepList.add(steps[i]);
            }
        }
        return new Pipeline<>(stepList);
    }
}
