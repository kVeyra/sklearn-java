package org.sklearn.cluster;

import org.sklearn.math.Matrix;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Agglomerative hierarchical clustering.
 *
 * <p>Performs hierarchical clustering using a bottom-up approach:
 * each sample starts as its own cluster, and pairs of clusters are
 * merged iteratively based on the linkage criterion.
 *
 * <p>Mirrors {@code sklearn.cluster.AgglomerativeClustering}.
 */
public class AgglomerativeClustering {

    private int nClusters;
    private String linkage;
    private boolean fitted;
    private int[] labels;
    private int nFeatures;

    /**
     * Create AgglomerativeClustering.
     *
     * @param nClusters number of clusters
     * @param linkage   linkage criterion: "ward", "complete", "average", or "single"
     */
    public AgglomerativeClustering(int nClusters, String linkage) {
        if (nClusters < 1) throw new IllegalArgumentException("nClusters must be >= 1");
        if (!linkage.equals("ward") && !linkage.equals("complete")
            && !linkage.equals("average") && !linkage.equals("single")) {
            throw new IllegalArgumentException("Unknown linkage: " + linkage);
        }
        this.nClusters = nClusters;
        this.linkage = linkage;
    }

    public int[] fitPredict(Matrix X) {
        fit(X);
        return labels;
    }

    public AgglomerativeClustering fit(Matrix X) {
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        nFeatures = X.cols();

        if (nClusters > n) {
            throw new IllegalArgumentException("nClusters cannot be greater than n_samples");
        }

        double[][] dist = computePairwiseDistances(X);
        int[] clusterIds = new int[n];
        for (int i = 0; i < n; i++) clusterIds[i] = i;

        int currentN = n;
        while (currentN > nClusters) {
            int bestI = -1, bestJ = -1;
            double bestDist = Double.MAX_VALUE;

            for (int i = 0; i < n; i++) {
                if (clusterIds[i] != i) continue;
                for (int j = i + 1; j < n; j++) {
                    if (clusterIds[j] != j) continue;
                    double d = linkageDist(i, j, clusterIds, dist, n);
                    if (d < bestDist) {
                        bestDist = d;
                        bestI = i;
                        bestJ = j;
                    }
                }
            }

            if (bestI >= 0) {
                int mergeId = Math.min(bestI, bestJ);
                int removedId = Math.max(bestI, bestJ);
                for (int k = 0; k < n; k++) {
                    if (clusterIds[k] == removedId) {
                        clusterIds[k] = mergeId;
                    }
                }
                currentN--;
            } else {
                break;
            }
        }

        Map<Integer, Integer> labelMap = new LinkedHashMap<>();
        int label = 0;
        labels = new int[n];
        for (int i = 0; i < n; i++) {
            int id = clusterIds[i];
            if (!labelMap.containsKey(id)) {
                labelMap.put(id, label++);
            }
            labels[i] = labelMap.get(id);
        }

        fitted = true;
        return this;
    }

    private double linkageDist(int i, int j, int[] clusterIds, double[][] dist, int n) {
        switch (linkage) {
            case "single": {
                double min = Double.MAX_VALUE;
                for (int a = 0; a < n; a++) {
                    if (clusterIds[a] != i) continue;
                    for (int b = 0; b < n; b++) {
                        if (clusterIds[b] != j) continue;
                        min = Math.min(min, dist[a][b]);
                    }
                }
                return min;
            }
            case "complete": {
                double max = -Double.MAX_VALUE;
                for (int a = 0; a < n; a++) {
                    if (clusterIds[a] != i) continue;
                    for (int b = 0; b < n; b++) {
                        if (clusterIds[b] != j) continue;
                        max = Math.max(max, dist[a][b]);
                    }
                }
                return max;
            }
            case "average": {
                double sum = 0;
                int count = 0;
                for (int a = 0; a < n; a++) {
                    if (clusterIds[a] != i) continue;
                    for (int b = 0; b < n; b++) {
                        if (clusterIds[b] != j) continue;
                        sum += dist[a][b];
                        count++;
                    }
                }
                return count > 0 ? sum / count : Double.MAX_VALUE;
            }
            case "ward": {
                double sum = 0;
                int count = 0;
                for (int a = 0; a < n; a++) {
                    if (clusterIds[a] != i) continue;
                    for (int b = 0; b < n; b++) {
                        if (clusterIds[b] != j) continue;
                        sum += dist[a][b] * dist[a][b];
                        count++;
                    }
                }
                return count > 0 ? Math.sqrt(sum / count) : Double.MAX_VALUE;
            }
            default:
                return Double.MAX_VALUE;
        }
    }

    private double[][] computePairwiseDistances(Matrix X) {
        int n = X.rows();
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < nFeatures; k++) {
                    double diff = X.get(i, k) - X.get(j, k);
                    sum += diff * diff;
                }
                d[i][j] = Math.sqrt(sum);
                d[j][i] = d[i][j];
            }
        }
        return d;
    }

    public int[] getLabels() { return labels; }

    public boolean isFitted() { return fitted; }

    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_clusters", nClusters);
        params.put("linkage", linkage);
        return Collections.unmodifiableMap(params);
    }
}
