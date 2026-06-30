package org.sklearn.core;

/**
 * Interface for estimators that make predictions.
 *
 * <p>Mirrors scikit-learn's {@code PredictorMixin}. Implementing classes
 * must be fitted before {@link #predict} or {@link #score} is called.
 *
 * @param <D> the input data type
 * @param <T> the target/label type used during training
 * @param <P> the prediction output type
 */
public interface Predictor<D, T, P> extends Estimator<D, T> {

    /**
     * Predict target values for samples in X.
     *
     * @param X samples to predict, shape (n_samples, n_features)
     * @return predicted values, shape (n_samples,)
     */
    P predict(D X);

    /**
     * Compute the coefficient of determination R² of the prediction.
     *
     * <p>For classifiers, this returns the mean accuracy.
     * For regressors, this returns the R² score.
     *
     * @param X test samples, shape (n_samples, n_features)
     * @param y true values, shape (n_samples,)
     * @return score (1.0 is perfect prediction)
     */
    double score(D X, T y);
}
