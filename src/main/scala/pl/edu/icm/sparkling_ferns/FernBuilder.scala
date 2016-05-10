package pl.edu.icm.sparkling_ferns

import org.apache.spark.mllib.linalg.Vector

/**
 * @author Mateusz Fedoryszak (mfedoryszak@gmail.com)
 */
case class FernBuilder(featureIndices: List[Int], thresholds: Map[Int, Double], categoricalFeaturesInfo: Map[Int, Int]) extends Serializable {
  val binarisers = Fern.sampleBinarisersPresetThresholds(thresholds, featureIndices, categoricalFeaturesInfo)

  def toCombinationIndex(featureVector: Vector): Int = {
    val features = featureVector.toArray
    val selected = featureIndices.map(features)

    Fern.toPointIndex(selected, binarisers)
  }

  def build(counts: Array[((Double, Int), Long)], labels: Array[Double]): FernModel = {
    val numFeatures = featureIndices.length
    val numDistinctPoints = 1 << numFeatures
    val scores = Fern.computeScores(counts, numDistinctPoints, labels)
    new FernModel(labels, featureIndices, binarisers, scores)
  }
}
