package org.sklearn.pipeline;

import org.sklearn.core.Estimator;
import org.sklearn.core.Predictor;
import org.sklearn.core.Transformer;

import java.util.*;

/**
 * A sequential pipeline of transformers followed by a final estimator.
 *
 * <p>Mirrors scikit-learn's {@code Pipeline}. Intermediate steps must be
 * {@link Transformer} instances; the final step must be an {@link Estimator}.
 *
 * <p>Usage:
 * <pre>{@code
 * Pipeline<Matrix, Vector, int[]> pipeline = new Pipeline<>(
 *     List.of(new StandardScaler()),
 *     new LogisticRegression()
 * );
 * pipeline.fit(X, y);
 * int[] predictions = pipeline.predict(X_test);
 * }</pre>
 *
 * @param <D> the data type flowing through the pipeline
 * @param <T> the target type
 * @param <P> the prediction type (only relevant if the final step is a Predictor)
 */
public class Pipeline<D, T, P> implements Estimator<D, T> {

    private final List<Transformer<D, ? super T>> transformers;
    private final Estimator<D, T> finalEstimator;

    /**
     * Create a pipeline with the given named steps.
     *
     * @param steps list of steps where all but the last must be Transformer,
     *              and the last must be an Estimator
     */
    public Pipeline(List<?> steps) {
        if (steps == null || steps.isEmpty()) {
            throw new IllegalArgumentException("Pipeline must have at least one step");
        }
        this.transformers = new ArrayList<>();
        for (int i = 0; i < steps.size() - 1; i++) {
            Object step = steps.get(i);
            if (!(step instanceof Transformer)) {
                throw new IllegalArgumentException(
                    "Step " + i + " must be a Transformer, got: " + step.getClass());
            }
            @SuppressWarnings("unchecked")
            Transformer<D, ? super T> t = (Transformer<D, ? super T>) step;
            this.transformers.add(t);
        }
        Object last = steps.get(steps.size() - 1);
        if (!(last instanceof Estimator)) {
            throw new IllegalArgumentException(
                "Final step must be an Estimator, got: " + last.getClass());
        }
        @SuppressWarnings("unchecked")
        Estimator<D, T> e = (Estimator<D, T>) last;
        this.finalEstimator = e;
    }

    @Override
    public Pipeline<D, T, P> fit(D X, T y) {
        D current = X;
        for (Transformer<D, ? super T> transformer : transformers) {
            transformer.fit(current, y);
            current = transformer.transform(current);
        }
        finalEstimator.fit(current, y);
        return this;
    }

    /**
     * Transform data through all transformer steps.
     *
     * @param X input data
     * @return data after all transformations
     */
    public D transform(D X) {
        D current = X;
        for (Transformer<D, ? super T> transformer : transformers) {
            current = transformer.transform(current);
        }
        return current;
    }

    /**
     * Transform data and then predict using the final estimator.
     *
     * @param X input data
     * @return predictions from the final estimator
     */
    @SuppressWarnings("unchecked")
    public P predict(D X) {
        D transformed = X;
        for (Transformer<D, ? super T> transformer : transformers) {
            transformed = transformer.transform(transformed);
        }
        if (finalEstimator instanceof Predictor) {
            return ((Predictor<D, T, P>) finalEstimator).predict(transformed);
        }
        throw new IllegalStateException(
            "Final estimator is not a Predictor: " + finalEstimator.getClass());
    }

    /**
     * Score the pipeline on test data.
     *
     * @param X test samples
     * @param y true targets
     * @return score
     */
    @SuppressWarnings("unchecked")
    public double score(D X, T y) {
        D transformed = X;
        for (Transformer<D, ? super T> transformer : transformers) {
            transformed = transformer.transform(transformed);
        }
        if (finalEstimator instanceof Predictor) {
            return ((Predictor<D, T, P>) finalEstimator).score(transformed, y);
        }
        throw new IllegalStateException(
            "Final estimator is not a Predictor: " + finalEstimator.getClass());
    }

    /**
     * Return the final estimator in the pipeline.
     *
     * @return the last step
     */
    public Estimator<D, T> getFinalEstimator() {
        return finalEstimator;
    }

    /**
     * Return the list of transformer steps.
     *
     * @return unmodifiable list of transformers
     */
    public List<Transformer<D, ? super T>> getTransformers() {
        return Collections.unmodifiableList(transformers);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        for (int i = 0; i < transformers.size(); i++) {
            params.put("step" + i, transformers.get(i).getParameters());
        }
        params.put("final", finalEstimator.getParameters());
        return Collections.unmodifiableMap(params);
    }
}
