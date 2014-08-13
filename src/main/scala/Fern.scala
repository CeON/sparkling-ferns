import breeze.numerics.log
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

import scala.util.Random

class FernModel (
    val labels: Array[Double],
    val featureIndices: List[Int],
    val scores: Array[Array[Double]]) extends ClassificationModel with Serializable {
  override def predict(testData: RDD[Vector]): RDD[Double] = testData.map(predict)

  override def predict(testData: Vector): Double = {
    val features = testData.toArray
    val selected = featureIndices.map(features)

    val pointIdx = Fern.toPointIndex(selected)

    val labelIdx = (0 until labels.length).maxBy(i => scores(i)(pointIdx))

    labels(labelIdx)
  }

  def scores(testData: Vector): Array[Double] = {
    val features = testData.toArray
    val selected = featureIndices.map(features)

    val pointIdx = Fern.toPointIndex(selected)

    (0 until labels.length).map(i => scores(i)(pointIdx)).toArray
  }
}

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class Fern {
  def run(data: RDD[LabeledPoint], featureIndices: List[Int]): FernModel = {
    val numFeatures = featureIndices.length

    val converted = data.map { p =>
      val features = p.features.toArray
      val selected = featureIndices.map(features)

      (p.label, Fern.toPointIndex(selected))
    }

    val aggregated = converted.groupBy(identity).map(x => (x._1, x._2.size)).collect()
    
    val labels = aggregated.map(_._1._1).distinct
    val labelsRev = labels.toList.zipWithIndex.toMap
    val numLabels = labels.length
    val numDistinctPoints = 1 << numFeatures

    val objectsInLeafPerLabel = Array.fill[Long](numLabels, numDistinctPoints)(1)
    val objectsInLeaf = Array.fill[Long](numDistinctPoints)(0)
    val objectsPerLabel = Array.fill[Long](numLabels)(0)

    aggregated.foreach{ case ((label, pointIdx), count) =>
      val labelIdx = labelsRev(label)
      objectsInLeafPerLabel(labelIdx)(pointIdx) += count
      objectsInLeaf(pointIdx) += count
      objectsPerLabel(labelIdx) += count
    }

    val numSamples = objectsPerLabel.sum

    val scores = Array.tabulate[Double](numLabels, numDistinctPoints) { case (label, pointIdx) => log(
      (objectsInLeafPerLabel(label)(pointIdx) + 1).toDouble/(objectsInLeaf(pointIdx) + numLabels)
      *
      (numSamples + numLabels).toDouble/(objectsPerLabel(label) + 1)
      )}

    new FernModel(labels, featureIndices, scores)
  }
}

object Fern {
  def toPointIndex(list: List[Double]): Int = {
    def toVal(d: Double): Int =
      if (d > 0)
        1
      else
        0

    def helper(list: List[Double], acc: Int): Int = list match {
      case Nil => acc
      case h::t => helper(t, 2*acc + toVal(h))
    }

    helper(list, 0)
  }

  def train(input: RDD[LabeledPoint], numFeatures: Int): FernModel = {
    val allFeaturesNo = input.first().features.size
    val featureIndices = Random.shuffle(0 until allFeaturesNo toList).take(numFeatures).sorted

    new Fern().run(input, featureIndices)
  }
  
  def train(input: RDD[LabeledPoint], featureIndices: List[Int]): FernModel = {
    new Fern().run(input, featureIndices)
  }
}