package org.sklearn.metrics;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.*;

/**
 * Clustering metrics: silhouette score, adjusted Rand index, etc.
 *
 * <p>Mirrors {@code sklearn.metrics.cluster} functions.
 */
public final class ClusteringMetrics {

    private ClusteringMetrics() {
    }

    /**
     * Compute the mean silhouette coefficient of all samples.
     */
    public static double silhouetteScore(Matrix X, Vector labels) {
        int n = X.rows();
        double total = 0;
        for (int i = 0; i < n; i++) {
            total += silhouetteSample(X, labels, i);
        }
        return total / n;
    }

    private static double silhouetteSample(Matrix X, Vector labels, int i) {
        int n = X.rows();
        int label = (int) labels.get(i);

        double a = 0;
        int aCount = 0;
        double b = Double.MAX_VALUE;
        Set<Integer> otherLabels = new HashSet<>();

        for (int j = 0; j < n; j++) {
            if (j == i) {
                continue;
            }
            otherLabels.add((int) labels.get(j));
        }
        otherLabels.remove(label);

        for (int j = 0; j < n; j++) {
            if (j == i) {
                continue;
            }
            double dist = euclidean(X, i, j);
            if ((int) labels.get(j) == label) {
                a += dist;
                aCount++;
            }
        }
        a = aCount > 0 ? a / aCount : 0;

        for (int other : otherLabels) {
            double sum = 0;
            int count = 0;
            for (int j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                if ((int) labels.get(j) == other) {
                    sum += euclidean(X, i, j);
                    count++;
                }
            }
            double meanDist = count > 0 ? sum / count : 0;
            b = Math.min(b, meanDist);
        }

        double max = Math.max(a, b);
        return max > 0 ? (b - a) / max : 0;
    }

    private static double euclidean(Matrix X, int i, int j) {
        double sum = 0;
        for (int k = 0; k < X.cols(); k++) {
            double diff = X.get(i, k) - X.get(j, k);
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * Compute the adjusted Rand index.
     */
    public static double adjustedRandScore(Vector labelsTrue, Vector labelsPred) {
        checkLengths(labelsTrue, labelsPred);
        int n = labelsTrue.size();

        int[][] cm = contingencyMatrix(labelsTrue, labelsPred);
        int nRows = cm.length;
        int nCols = cm[0].length;

        double sumComb = 0;
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                sumComb += comb2(cm[i][j]);
            }
        }

        double sumRows = 0;
        for (int i = 0; i < nRows; i++) {
            int rowSum = 0;
            for (int j = 0; j < nCols; j++) {
                rowSum += cm[i][j];
            }
            sumRows += comb2(rowSum);
        }

        double sumCols = 0;
        for (int j = 0; j < nCols; j++) {
            int colSum = 0;
            for (int i = 0; i < nRows; i++) {
                colSum += cm[i][j];
            }
            sumCols += comb2(colSum);
        }

        double totalComb = comb2(n);
        double expected = sumRows * sumCols / totalComb;
        double maxPossible = (sumRows + sumCols) / 2.0;

        double num = sumComb - expected;
        double den = maxPossible - expected;
        return Math.abs(den) < 1e-15 ? 0.0 : num / den;
    }

    private static int[][] contingencyMatrix(Vector a, Vector b) {
        int n = a.size();
        Set<Integer> aLabels = new LinkedHashSet<>();
        Set<Integer> bLabels = new LinkedHashSet<>();
        for (int i = 0; i < n; i++) {
            aLabels.add((int) a.get(i));
            bLabels.add((int) b.get(i));
        }
        int[] aArr = aLabels.stream().mapToInt(Integer::intValue).toArray();
        int[] bArr = bLabels.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(aArr);
        Arrays.sort(bArr);

        int[][] cm = new int[aArr.length][bArr.length];
        Map<Integer, Integer> aMap = new HashMap<>();
        Map<Integer, Integer> bMap = new HashMap<>();
        for (int i = 0; i < aArr.length; i++) {
            aMap.put(aArr[i], i);
        }
        for (int j = 0; j < bArr.length; j++) {
            bMap.put(bArr[j], j);
        }

        for (int i = 0; i < n; i++) {
            int ai = aMap.get((int) a.get(i));
            int bj = bMap.get((int) b.get(i));
            cm[ai][bj]++;
        }
        return cm;
    }

    private static double comb2(int x) {
        return x < 2 ? 0 : (double) x * (x - 1) / 2;
    }

    /**
     * Compute mutual information score.
     */
    public static double mutualInfoScore(Vector labelsTrue, Vector labelsPred) {
        checkLengths(labelsTrue, labelsPred);
        int n = labelsTrue.size();
        int[][] cm = contingencyMatrix(labelsTrue, labelsPred);
        int nRows = cm.length;
        int nCols = cm[0].length;

        double mi = 0;
        for (int i = 0; i < nRows; i++) {
            int rowSum = 0;
            for (int j = 0; j < nCols; j++) {
                rowSum += cm[i][j];
            }
            for (int j = 0; j < nCols; j++) {
                if (cm[i][j] == 0) {
                    continue;
                }
                int colSum = 0;
                for (int k = 0; k < nRows; k++) {
                    colSum += cm[k][j];
                }
                double pxy = (double) cm[i][j] / n;
                double px = (double) rowSum / n;
                double py = (double) colSum / n;
                mi += pxy * Math.log(pxy / (px * py));
            }
        }
        return mi;
    }

    /**
     * Compute adjusted mutual information score.
     */
    public static double adjustedMutualInfoScore(Vector labelsTrue, Vector labelsPred) {
        double mi = mutualInfoScore(labelsTrue, labelsPred);
        if (mi == 0) {
            return 0;
        }
        double emi = expectedMutualInfo(labelsTrue, labelsPred);
        double hTrue = entropy(labelsTrue);
        double hPred = entropy(labelsPred);
        double maxH = Math.max(hTrue, hPred);
        double den = maxH - emi;
        return Math.abs(den) < 1e-15 ? 0 : (mi - emi) / den;
    }

    private static double expectedMutualInfo(Vector a, Vector b) {
        int n = a.size();
        int[][] cm = contingencyMatrix(a, b);
        int nRows = cm.length;
        int nCols = cm[0].length;
        int[] rowSums = new int[nRows];
        int[] colSums = new int[nCols];
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                rowSums[i] += cm[i][j];
                colSums[j] += cm[i][j];
            }
        }
        double emi = 0;
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                if (cm[i][j] == 0) {
                    continue;
                }
                double num = 0;
                double aMax = Math.max(1, rowSums[i] + colSums[j] - n);
                double aMin = Math.min(rowSums[i], colSums[j]);
                for (int aij = (int) Math.ceil(aMax); aij <= (int) Math.floor(aMin); aij++) {
                    num += logTerm(aij, rowSums[i], colSums[j], n);
                }
                emi += num;
            }
        }
        return emi;
    }

    private static double logTerm(int a, int rowSum, int colSum, int n) {
        double term = (double) a / n * Math.log((double) a * n / (rowSum * colSum));
        return term * comb2(rowSum) * comb2(n - rowSum) * comb2(colSum) * comb2(n - colSum)
            / (comb2(n) * comb2(a) * comb2(rowSum - a) * comb2(colSum - a) * comb2(n - rowSum - colSum + a));
    }

    private static double entropy(Vector labels) {
        int n = labels.size();
        Map<Integer, Integer> counts = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int l = (int) labels.get(i);
            counts.put(l, counts.getOrDefault(l, 0) + 1);
        }
        double h = 0;
        for (int c : counts.values()) {
            double p = (double) c / n;
            h -= p * Math.log(p);
        }
        return h;
    }

    public static double normalizedMutualInfoScore(Vector labelsTrue, Vector labelsPred) {
        double mi = mutualInfoScore(labelsTrue, labelsPred);
        if (mi == 0) {
            return 0;
        }
        double hTrue = entropy(labelsTrue);
        double hPred = entropy(labelsPred);
        double avgH = (hTrue + hPred) / 2.0;
        return avgH > 0 ? mi / avgH : 0;
    }

    public static double homogeneityScore(Vector labelsTrue, Vector labelsPred) {
        double mi = mutualInfoScore(labelsTrue, labelsPred);
        double hTrue = entropy(labelsTrue);
        return hTrue > 0 ? mi / hTrue : 1.0;
    }

    public static double completenessScore(Vector labelsTrue, Vector labelsPred) {
        double mi = mutualInfoScore(labelsTrue, labelsPred);
        double hPred = entropy(labelsPred);
        return hPred > 0 ? mi / hPred : 1.0;
    }

    public static double vMeasureScore(Vector labelsTrue, Vector labelsPred) {
        double h = homogeneityScore(labelsTrue, labelsPred);
        double c = completenessScore(labelsTrue, labelsPred);
        return h + c > 0 ? 2 * h * c / (h + c) : 0;
    }

    public static double randScore(Vector labelsTrue, Vector labelsPred) {
        checkLengths(labelsTrue, labelsPred);
        int n = labelsTrue.size();
        int[][] cm = contingencyMatrix(labelsTrue, labelsPred);
        int nRows = cm.length;
        int nCols = cm[0].length;
        double sumComb = 0;
        double sumRowComb = 0;
        double sumColComb = 0;
        for (int i = 0; i < nRows; i++) {
            int rowSum = 0;
            for (int j = 0; j < nCols; j++) {
                sumComb += comb2(cm[i][j]);
                rowSum += cm[i][j];
            }
            sumRowComb += comb2(rowSum);
        }
        for (int j = 0; j < nCols; j++) {
            int colSum = 0;
            for (int i = 0; i < nRows; i++) {
                colSum += cm[i][j];
            }
            sumColComb += comb2(colSum);
        }
        double totalComb = comb2(n);
        return (2.0 * sumComb + totalComb - sumRowComb - sumColComb) / totalComb;
    }

    public static double fowlkesMallowsScore(Vector labelsTrue, Vector labelsPred) {
        checkLengths(labelsTrue, labelsPred);
        int[][] cm = contingencyMatrix(labelsTrue, labelsPred);
        int nRows = cm.length;
        int nCols = cm[0].length;

        double tk = 0;
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                tk += comb2(cm[i][j]);
            }
        }
        double pk = 0;
        for (int i = 0; i < nRows; i++) {
            int sum = 0;
            for (int j = 0; j < nCols; j++) {
                sum += cm[i][j];
            }
            pk += comb2(sum);
        }
        double qk = 0;
        for (int j = 0; j < nCols; j++) {
            int sum = 0;
            for (int i = 0; i < nRows; i++) {
                sum += cm[i][j];
            }
            qk += comb2(sum);
        }
        double den = Math.sqrt(pk * qk);
        return den > 0 ? tk / den : 0;
    }

    public static double calinskiHarabaszScore(Matrix X, Vector labels) {
        int n = X.rows();
        Set<Integer> unique = new HashSet<>();
        for (int i = 0; i < labels.size(); i++) {
            unique.add((int) labels.get(i));
        }
        int k = unique.size();
        if (k == 1) {
            return 0;
        }

        Vector overallMean = new Vector(X.cols());
        for (int j = 0; j < X.cols(); j++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += X.get(i, j);
            }
            overallMean.set(j, sum / n);
        }

        double bg = 0;
        double wg = 0;
        Set<Integer> uniqueLabels = new LinkedHashSet<>();
        for (int i = 0; i < n; i++) {
            uniqueLabels.add((int) labels.get(i));
        }

        for (int label : uniqueLabels) {
            List<Integer> members = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if ((int) labels.get(i) == label) {
                    members.add(i);
                }
            }
            int nk = members.size();
            Vector clusterMean = new Vector(X.cols());
            for (int j = 0; j < X.cols(); j++) {
                double sum = 0;
                for (int idx : members) {
                    sum += X.get(idx, j);
                }
                clusterMean.set(j, sum / nk);
            }

            for (int j = 0; j < X.cols(); j++) {
                bg += nk * Math.pow(clusterMean.get(j) - overallMean.get(j), 2);
            }

            for (int idx : members) {
                for (int j = 0; j < X.cols(); j++) {
                    wg += Math.pow(X.get(idx, j) - clusterMean.get(j), 2);
                }
            }
        }

        int d = X.cols();
        return (bg / (k - 1)) / (wg / (n - k));
    }

    private static void checkLengths(Vector a, Vector b) {
        if (a.size() != b.size()) {
            throw new IllegalArgumentException(
                "Vectors must have same length: " + a.size() + " vs " + b.size());
        }
    }
}
