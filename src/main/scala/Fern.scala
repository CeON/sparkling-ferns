import breeze.numerics.log
import breeze.stats.distributions.Poisson
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class FernModel (
    val labels: Array[Double],
    val featureIndices: List[Int],
    val thresholds: List[Double],
    val scores: Array[Array[Double]]) extends ClassificationModel with Serializable {
  override def predict(testData: RDD[Vector]): RDD[Double] = testData.map(predict)

  override def predict(testData: Vector): Double = {
    val labelIdx = (0 until labels.length).maxBy(scores(testData))

    labels(labelIdx)
  }

  /**
   * @return an array of scores for each label for a given Vector
   */
  def scores(testData: Vector): Array[Double] = {
    val features = testData.toArray
    val selected = featureIndices.map(features)

    val pointIdx = Fern.toPointIndex(selected, thresholds)

    (0 until labels.length).map(i => scores(i)(pointIdx)).toArray
  }

  /**
   * @return a list of ((correct label, predicted label), count)
   */
  def confusionMatrix(testData: RDD[LabeledPoint]): List[((Double, Double), Long)] = {
    testData.map(p => (p.label, predict(p.features))).countByValue().toList
  }
  
  def featureImportance(testData: RDD[LabeledPoint]): List[(Int, Double)] = {
    val labelsRev = labels.toList.zipWithIndex.toMap

    featureIndices.map { index =>
      val shuffled = Fern.shuffleFeatureValues(testData, index)
      val importance = shuffled.map{case (p, s) =>
        val labelIndex = labelsRev(p.label)
        scores(p.features)(labelIndex) - scores(s.features)(labelIndex)
      }.mean()
      (index, importance)
    }
  }
}

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
case class FernModelWithStats(model: FernModel,
                              oobConfusionMatrix: List[((Double, Double), Long)],
                              featureImportance: List[(Int, Double)])

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class Fern(val presetLabels: Option[Array[Double]] = None) {
  /**
   * Constructs a model using a whole data as a training sample.
   */
  def run(data: RDD[LabeledPoint], featureIndices: List[Int], thresholds: List[Double]): FernModel = {
    val numFeatures = featureIndices.length
    val numDistinctPoints = 1 << numFeatures
    val labels = presetLabels.getOrElse(data.map(_.label).distinct().collect())

    val converted = data.map { p =>
      val features = p.features.toArray
      val selected = featureIndices.map(features)

      (p.label, Fern.toPointIndex(selected, thresholds))
    }

    val scores = computeScores(converted, numDistinctPoints, labels)

    new FernModel(labels, featureIndices, thresholds, scores)
  }

  /**
   * Constructs a model and assesses it. Uses bagging to divide the data into training and validation set. Sampling
   * with replacement is simulated by sampling a number of occurrences for each element from a Poisson distribution with
   * lambda = 1.
   */
  def runAndAssess(data: RDD[LabeledPoint], featureIndices: List[Int], thresholds: List[Double]): FernModelWithStats = {
    val withMultipliers = data.map(x => (x, Poisson.distribution(1.0).draw()))

    val training = withMultipliers.flatMap{case (x, mul) => List.fill(mul)(x)}
    val oob = withMultipliers.filter(_._2 == 0).map(_._1)

    val model = run(training, featureIndices, thresholds)

    val confusionMatrix = model.confusionMatrix(oob)

    val featureImportance = model.featureImportance(oob)

    FernModelWithStats(model, confusionMatrix, featureImportance)
  }

  def computeScores(training: RDD[(Double, Int)], numDistinctPoints: Int, labels: Array[Double]) = {
    val aggregated = training.groupBy(identity).map(x => (x._1, x._2.size)).collect()

    val labelsRev = labels.toList.zipWithIndex.toMap
    val numLabels = labels.length

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

    scores
  }
}

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
object Fern {
  def toPointIndex(list: List[Double], thresholds: List[Double]): Int = {
    val binary = (list zip thresholds).map{ case (el, threshold) => if (el > threshold) 1 else 0}

    def helper(list: List[Int], acc: Int): Int = list match {
      case Nil => acc
      case h::t => helper(t, 2*acc + h)
    }

    helper(binary, 0)
  }

  /**
   * Continuous features are transformed into binary ones by sampling cut-off thresholds from the data. That's what this
   * method does.
   */
  def sampleThresholds(data: RDD[LabeledPoint], featureIndices: List[Int]): List[Double] = {
    data.map { p =>
      val features = p.features.toArray
      val selected = featureIndices.map(features)

      val marked = selected.map(x => List((x, Random.nextFloat())))
      marked
    }.reduce{(list1, list2) =>
      (list1 zip list2).map{ case (el1, el2) => (el1 ++ el2).sortBy(_._2).take(2)}
    }.map(list => list.unzip._1.sum / 2)
  }

  def sampleFeatureIndices(data: RDD[LabeledPoint], numFeatures: Int): List[Int] = {
    val allFeaturesNo = data.first().features.size
    Random.shuffle(0 until allFeaturesNo toList).take(numFeatures).sorted
  }

  def shuffleFeatureValues(data: RDD[LabeledPoint], featureIndex: Int): RDD[(LabeledPoint, LabeledPoint)] = {
    val indexed = data.zipWithIndex().map(_.swap)
    val permuted = data.map(_.features.toArray(featureIndex)).map(x => (Random.nextDouble(), x)).sortByKey().map(_._2)
      .zipWithIndex().map(_.swap)

    val results = indexed.join(permuted).map { case (_, (point, value)) =>
      val array = point.features.toArray.clone()
      array(featureIndex) = value
      (point, LabeledPoint(point.label, Vectors.dense(array)))
    }

    results
  }

  def train(input: RDD[LabeledPoint], numFeatures: Int, labels: Array[Double]): FernModel = {
    val featureIndices = sampleFeatureIndices(input, numFeatures)
    val thresholds = sampleThresholds(input, featureIndices)

    new Fern(Some(labels)).run(input, featureIndices, thresholds)
  }

  def trainAndAssess(input: RDD[LabeledPoint], numFeatures: Int, labels: Array[Double]): FernModelWithStats = {
    val featureIndices = sampleFeatureIndices(input, numFeatures)
    val thresholds = sampleThresholds(input, featureIndices)

    new Fern(Some(labels)).runAndAssess(input, featureIndices, thresholds)
  }

  def train(input: RDD[LabeledPoint], featureIndices: List[Int], labels: Array[Double]): FernModel =
    new Fern(Some(labels)).run(input, featureIndices, sampleThresholds(input, featureIndices))

  def train(input: RDD[LabeledPoint], numFeatures: Int, labels: Array[Double], thresholds: List[Double]): FernModel =
    new Fern(Some(labels)).run(input, sampleFeatureIndices(input, numFeatures), thresholds)

  def train(input: RDD[LabeledPoint], featureIndices: List[Int], labels: Array[Double], thresholds: List[Double]): FernModel =
    new Fern(Some(labels)).run(input, featureIndices, thresholds)

  def train(input: RDD[LabeledPoint], numFeatures: Int, thresholds: List[Double]): FernModel =
    new Fern(None).run(input, sampleFeatureIndices(input, numFeatures), thresholds)

  def train(input: RDD[LabeledPoint], featureIndices: List[Int], thresholds: List[Double]): FernModel =
    new Fern(None).run(input, featureIndices, thresholds)
}
