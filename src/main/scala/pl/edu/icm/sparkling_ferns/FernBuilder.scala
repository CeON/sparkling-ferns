package pl.edu.icm.sparkling_ferns

import breeze.numerics._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class FernBuilder(featureIndices: List[Int], binarisers: List[FeatureBinariser]) extends Serializable {
  def toIndex(featureVector: Vector): Int = {
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
