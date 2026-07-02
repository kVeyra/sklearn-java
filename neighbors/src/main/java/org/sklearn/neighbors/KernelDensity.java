package org.sklearn.neighbors;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Kernel density estimation using a fixed bandwidth.
 */
public class KernelDensity {

    private double bandwidth;
    private String kernel;
    private String algorithm;
    private String metric;
    private boolean fitted;
    private Matrix xTrain;
    private int nFeatures;
    private double logBandwidth;

    private static final double SQRT_2PI = Math.sqrt(2 * Math.PI);

    public KernelDensity() {
        this(1.0, "gaussian", "auto", "euclidean");
    }

    public KernelDensity(double bandwidth, String kernel, String algorithm, String metric) {
        this.bandwidth = bandwidth;
        this.kernel = kernel;
        this.algorithm = algorithm;
        this.metric = metric;
    }

    public KernelDensity fit(Matrix X) {
        Validation.checkMatrix(X, -1);
        this.xTrain = new Matrix(X);
        this.nFeatures = X.cols();
        this.logBandwidth = Math.log(bandwidth);
        this.fitted = true;
        return this;
    }

    public Vector scoreSamples(Matrix X) {
        Validation.checkFitted(fitted, "KernelDensity");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures + " got " + X.cols());
        }

        int n = X.rows();
        int nTrain = xTrain.rows();
        double[] scores = new double[n];

        double normalization = 0.0;
        if ("gaussian".equals(kernel)) {
            normalization = -0.5 * nFeatures * Math.log(2 * Math.PI) - nFeatures * logBandwidth;
        } else {
            normalization = -Math.log(nTrain) - nFeatures * logBandwidth;
            if ("tophat".equals(kernel)) {
                normalization += Math.log(1.0 / volumeUnitBall());
            } else if ("epanechnikov".equals(kernel)) {
                normalization += Math.log((nFeatures + 2.0) / (2.0 * volumeUnitBall()));
            }
        }

        for (int i = 0; i < n; i++) {
            double logSum = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < nTrain; j++) {
                double dist = 0.0;
                for (int f = 0; f < nFeatures; f++) {
                    double diff = X.get(i, f) - xTrain.get(j, f);
                    dist += diff * diff;
                }
                dist = Math.sqrt(dist);
                double u = dist / bandwidth;
                double kernelVal = kernelEval(u);

                if (kernelVal > 0) {
                    double logK = Math.log(kernelVal);
                    logSum = logSumExp(logSum, logK);
                }
            }
            scores[i] = normalization + logSum - Math.log(nTrain);
        }
        return new Vector(scores);
    }

    public double score(Matrix X) {
        Vector scores = scoreSamples(X);
        double sum = 0.0;
        for (int i = 0; i < scores.size(); i++) {
            sum += scores.get(i);
        }
        return sum / scores.size();
    }

    private double kernelEval(double u) {
        switch (kernel) {
            case "gaussian":
                return Math.exp(-0.5 * u * u) / Math.pow(SQRT_2PI, nFeatures);
            case "tophat":
                return u <= 1.0 ? 1.0 : 0.0;
            case "epanechnikov":
                return u <= 1.0 ? 0.75 * (1.0 - u * u) : 0.0;
            case "exponential":
                return Math.exp(-u);
            case "linear":
                return u <= 1.0 ? 1.0 - u : 0.0;
            case "cosine":
                return u <= 1.0 ? Math.cos(Math.PI * u / 2.0) : 0.0;
            default:
                return Math.exp(-0.5 * u * u) / Math.pow(SQRT_2PI, nFeatures);
        }
    }

    private double volumeUnitBall() {
        if (nFeatures == 1) {
            return 2.0;
        }
        if (nFeatures == 2) {
            return Math.PI;
        }
        double logVol = nFeatures * 0.5 * Math.log(Math.PI) - lgamma(nFeatures / 2.0 + 1.0);
        return Math.exp(logVol);
    }

    private double lgamma(double x) {
        return lgammaStirling(x);
    }

    private double lgammaStirling(double x) {
        double[] coef = {76.18009172947146, -86.50532032941677,
            24.01409824083091, -1.231739572450155,
            0.1208650973866179e-2, -0.5395239384953e-5};
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++) {
            y += 1.0;
            ser += coef[j] / y;
        }
        return -tmp + Math.log(2.5066282746310005 * ser / x);
    }

    private double logSumExp(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) {
            return b;
        }
        if (b == Double.NEGATIVE_INFINITY) {
            return a;
        }
        double max = Math.max(a, b);
        return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
    }

    public boolean isFitted() {
        return fitted;
    }

    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("bandwidth", bandwidth);
        p.put("kernel", kernel);
        p.put("algorithm", algorithm);
        p.put("metric", metric);
        return Collections.unmodifiableMap(p);
    }
}
