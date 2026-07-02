package org.sklearn.pipeline;

import org.junit.jupiter.api.Test;
import org.sklearn.preprocessing.StandardScaler;
import org.sklearn.linear_model.LinearRegression;
import static org.junit.jupiter.api.Assertions.*;

class PipelineUtilsTest {

    @Test
    void testMakePipeline() {
        StandardScaler scaler = new StandardScaler();
        LinearRegression lr = new LinearRegression();
        Pipeline<?, ?, ?> p = PipelineUtils.makePipeline(scaler, lr);
        assertNotNull(p);
    }
}
