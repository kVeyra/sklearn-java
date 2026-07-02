package org.sklearn.pipeline;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;

import java.util.*;

public class FeatureUnion<D, T> implements Transformer<D, T> {

    private final List<String> names;
    private final List<Transformer<D, T>> transformers;
    private boolean fitted;

    public FeatureUnion(List<String> names, List<Transformer<D, T>> transformers) {
        this.names = names;
        this.transformers = transformers;
    }

    @Override
    public FeatureUnion<D, T> fit(D X, T y) {
        for (Transformer<D, T> t : transformers) {
            t.fit(X, y);
        }
        fitted = true;
        return this;
    }

    @Override
    @SuppressWarnings("unchecked")
    public D transform(D X) {
        if (!(X instanceof Matrix)) {
            throw new IllegalArgumentException("FeatureUnion currently only supports Matrix input");
        }
        Matrix mx = (Matrix) X;
        List<Matrix> parts = new ArrayList<>();
        for (Transformer<D, T> t : transformers) {
            parts.add((Matrix) t.transform(X));
        }
        int totalCols = 0;
        for (Matrix p : parts) totalCols += p.cols();
        Matrix result = new Matrix(mx.rows(), totalCols);
        int colOffset = 0;
        for (Matrix p : parts) {
            for (int i = 0; i < mx.rows(); i++) {
                for (int j = 0; j < p.cols(); j++) {
                    result.set(i, colOffset + j, p.get(i, j));
                }
            }
            colOffset += p.cols();
        }
        return (D) result;
    }

    @Override
    public D inverseTransform(D X) {
        throw new UnsupportedOperationException("FeatureUnion does not support inverseTransform");
    }

    @Override
    @SuppressWarnings("unchecked")
    public D fitTransform(D X, T y) {
        fit(X, y);
        return transform(X);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("names", names);
        p.put("transformers", transformers);
        return Collections.unmodifiableMap(p);
    }
}
