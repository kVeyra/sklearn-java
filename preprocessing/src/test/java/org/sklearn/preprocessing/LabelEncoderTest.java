package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class LabelEncoderTest {

    @Test
    void testBasicEncode() {
        LabelEncoder encoder = new LabelEncoder();
        int[] y = {2, 1, 3};
        encoder.fit(y, null);

        assertTrue(encoder.isFitted());

        int[] encoded = encoder.transform(y);
        assertArrayEquals(new int[]{1, 0, 2}, encoded);

        int[] classes = (int[]) encoder.getParameters().get("classes");
        assertArrayEquals(new int[]{1, 2, 3}, classes);
    }

    @Test
    void testInverseTransform() {
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(new int[]{2, 1, 3}, null);

        int[] decoded = encoder.inverseTransform(new int[]{1, 0, 2});
        assertArrayEquals(new int[]{2, 1, 3}, decoded);
    }

    @Test
    void testFitTransformRoundtrip() {
        LabelEncoder encoder = new LabelEncoder();
        int[] y = {2, 1, 3};

        int[] encoded = encoder.fitTransform(y, null);

        assertTrue(encoder.isFitted());
        assertArrayEquals(new int[]{1, 0, 2}, encoded);

        int[] decoded = encoder.inverseTransform(encoded);
        assertArrayEquals(y, decoded);
    }

    @Test
    void testValueErrorForUnseenLabel() {
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(new int[]{1, 2, 3}, null);

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
            () -> encoder.transform(new int[]{4}));
        assertTrue(ex.getMessage().contains("4"));
    }

    @Test
    void testEmptyInput() {
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(new int[]{}, null);

        assertTrue(encoder.isFitted());
        assertArrayEquals(new int[0], (int[]) encoder.getParameters().get("classes"));

        int[] encoded = encoder.transform(new int[]{});
        assertEquals(0, encoded.length);

        int[] decoded = encoder.inverseTransform(new int[]{});
        assertEquals(0, decoded.length);
    }

    @Test
    void testSingleClass() {
        LabelEncoder encoder = new LabelEncoder();
        int[] y = {5, 5, 5};
        encoder.fit(y, null);

        Map<String, Object> params = encoder.getParameters();
        assertArrayEquals(new int[]{5}, (int[]) params.get("classes"));

        int[] encoded = encoder.transform(y);
        assertArrayEquals(new int[]{0, 0, 0}, encoded);

        int[] decoded = encoder.inverseTransform(encoded);
        assertArrayEquals(y, decoded);
    }
}
