package org.sklearn.cluster;

import org.sklearn.math.Matrix;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Mean shift clustering using a flat kernel.
 *
 * <p>For each sample, iteratively shifts toward the mean of points
 * within bandwidth radius until convergence, then merges nearby
 * modes.
 *
 * <p>Mirrors {@code sklearn.cluster.MeanShift}.
 */
public class MeanShift {

    private double bandwidth;
    private int maxIter;
    private boolean fitted;
    private int[] labels;
    private Matrix clusterCenters;
    private int nFeatures;

    /**
     * Create MeanShift.
     *
     * @param bandwidth bandwidth radius (if &le; 0, estimated from data)
     * @param maxIter   maximum iterations per seed
     */
    public MeanShift(double bandwidth, int maxIter) {
        this.bandwidth = bandwidth;
        this.maxIter = maxIter;
    }

    public int[] fitPredict(Matrix X) {
        fit(X);
        return labels;
    }

    public MeanShift fit(Matrix X) {
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        nFeatures = X.cols();

        double bw = bandwidth > 0 ? bandwidth : estimateBandwidth(X);
        double tol = 1e-3;

        double[][] shifted = new double[n][nFeatures];
        for (int i = 0; i < n; i++) {
            double[] point = new double[nFeatures];
            for (int j = 0; j < nFeatures; j++) point[j] = X.get(i, j);

            for (int iter = 0; iter < maxIter; iter++) {
                double sumWeight = 0;
                double[] sum = new double[nFeatures];
                for (int k = 0; k < n; k++) {
                    double dist = 0;
                    for (int j = 0; j < nFeatures; j++) {
                        double diff = point[j] - X.get(k, j);
                        dist += diff * diff;
                    }
                    dist = Math.sqrt(dist);
                    if (dist <= bw) {
                        double w = 1.0;
                        sumWeight += w;
                        for (int j = 0; j < nFeatures; j++) {
                            sum[j] += w * X.get(k, j);
                        }
                    }
                }

                if (sumWeight == 0) break;

                double[] newPoint = new double[nFeatures];
                double shift = 0;
                for (int j = 0; j < nFeatures; j++) {
                    newPoint[j] = sum[j] / sumWeight;
                    shift += (newPoint[j] - point[j]) * (newPoint[j] - point[j]);
                }
                shift = Math.sqrt(shift);
                point = newPoint;

                if (shift < tol) break;
            }

            shifted[i] = point;
        }

        List<double[]> modes = new ArrayList<>();
        List<Integer> modeCounts = new ArrayList<>();
        double mergeDist = bw / 2.0;

        for (int i = 0; i < n; i++) {
            boolean merged = false;
            for (int m = 0; m < modes.size(); m++) {
                double dist = 0;
                for (int j = 0; j < nFeatures; j++) {
                    double diff = shifted[i][j] - modes.get(m)[j];
                    dist += diff * diff;
                }
                dist = Math.sqrt(dist);
                if (dist < mergeDist) {
                    int total = modeCounts.get(m) + 1;
                    double[] avg = new double[nFeatures];
                    for (int j = 0; j < nFeatures; j++) {
                        avg[j] = (modes.get(m)[j] * modeCounts.get(m) + shifted[i][j]) / total;
                    }
                    modes.set(m, avg);
                    modeCounts.set(m, total);
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                modes.add(shifted[i].clone());
                modeCounts.add(1);
            }
        }

        int nClusters = modes.size();
        clusterCenters = new Matrix(nClusters, nFeatures);
        for (int m = 0; m < nClusters; m++) {
            for (int j = 0; j < nFeatures; j++) {
                clusterCenters.set(m, j, modes.get(m)[j]);
            }
        }

        labels = new int[n];
        for (int i = 0; i < n; i++) {
            int best = 0;
            double bestDist = Double.MAX_VALUE;
            for (int m = 0; m < nClusters; m++) {
                double dist = 0;
                for (int j = 0; j < nFeatures; j++) {
                    double diff = X.get(i, j) - clusterCenters.get(m, j);
                    dist += diff * diff;
                }
                if (dist < bestDist) {
                    bestDist = dist;
                    best = m;
                }
            }
            labels[i] = best;
        }

        fitted = true;
        return this;
    }

    private double estimateBandwidth(Matrix X) {
        int n = X.rows();
        double[] dists = new double[n * (n - 1) / 2];
        int idx = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dist = 0;
                for (int k = 0; k < nFeatures; k++) {
                    double diff = X.get(i, k) - X.get(j, k);
                    dist += diff * diff;
                }
                dists[idx++] = Math.sqrt(dist);
            }
        }
        Arrays.sort(dists);
        int medianIdx = dists.length / 2;
        return dists.length > 0 ? dists[medianIdx] : 1.0;
    }

    public int[] getLabels() { return labels; }

    public Matrix getClusterCenters() { return clusterCenters; }

    public boolean isFitted() { return fitted; }

    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("bandwidth", bandwidth);
        params.put("max_iter", maxIter);
        return Collections.unmodifiableMap(params);
    }
}
