package pl.edu.icm.sparkling_ferns

import breeze.numerics.log
import breeze.stats.distributions.Poisson
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.immutable.BitSet
import scala.util.Random

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class FernModel (
    val labels: Array[Double],
    val featureIndices: List[Int],
    val binarisers: List[FeatureBinariser],
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

    val pointIdx = Fern.toPointIndex(selected, binarisers)

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
  def run(data: RDD[LabeledPoint], featureIndices: List[Int], binarisers: List[FeatureBinariser]): FernModel = {
    val numFeatures = featureIndices.length
    val numDistinctPoints = 1 << numFeatures
    val labels = presetLabels.getOrElse(util.extractLabels(data))

    val converted = data.map { p =>
      val features = p.features.toArray
      val selected = featureIndices.map(features)

      (p.label, Fern.toPointIndex(selected, binarisers))
    }

    val scores = computeScores(converted, numDistinctPoints, labels)

    new FernModel(labels, featureIndices, binarisers, scores)
  }

  /**
   * Constructs a model and assesses it. Uses bagging to divide the data into training and validation set. Sampling
   * with replacement is simulated by sampling a number of occurrences for each element from a Poisson distribution with
   * lambda = 1.
   */
  def runAndAssess(data: RDD[LabeledPoint], featureIndices: List[Int], binarisers: List[FeatureBinariser]): FernModelWithStats = {
    val withMultipliers = data.map(x => (x, Poisson.distribution(1.0).draw()))

    val training = withMultipliers.flatMap{case (x, mul) => List.fill(mul)(x)}
    val oob = withMultipliers.filter(_._2 == 0).map(_._1)

    val model = run(training, featureIndices, binarisers)

    val confusionMatrix = model.confusionMatrix(oob)

    val featureImportance = model.featureImportance(oob)

    FernModelWithStats(model, confusionMatrix, featureImportance)
  }

  def computeScores(training: RDD[(Double, Int)], numDistinctPoints: Int, labels: Array[Double]) = {
    val aggregated = training.groupBy(identity).map(x => (x._1, x._2.size.toLong)).collect()
    Fern.computeScores(aggregated, numDistinctPoints, labels)
  }
}

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
object Fern {
  def toPointIndex(list: List[Double], binarisers: List[FeatureBinariser]): Int = {
    val binary = (list zip binarisers).map{ case (el, binariser) => if (binariser(el)) 1 else 0}

    def helper(list: List[Int], acc: Int): Int = list match {
      case Nil => acc
      case h::t => helper(t, 2*acc + h)
    }

    helper(binary, 0)
  }

  def sampleBinarisers(data: RDD[LabeledPoint], featureIndices: List[Int], categoricalFeaturesInfo: Map[Int, Int]): List[FeatureBinariser] = {
    val continuousFeatureIndices = featureIndices.filterNot(categoricalFeaturesInfo.contains)
    val thresholds = sampleThresholds(data, continuousFeatureIndices)

    sampleBinarisersPresetThresholds(thresholds, featureIndices, categoricalFeaturesInfo)
  }

  def sampleBinarisersPresetThresholds(thresholds: Map[Int, Double], featureIndices: List[Int], categoricalFeaturesInfo: Map[Int, Int]): List[FeatureBinariser] = {
    val selectedCategoricalFeaturesInfo = categoricalFeaturesInfo.filterKeys(featureIndices.contains)
    val subsets = sampleSubsets(selectedCategoricalFeaturesInfo)

    featureIndices.map {idx =>
      if (categoricalFeaturesInfo.contains(idx))
        new CategoricalFeatureBinariser(subsets(idx))
      else
        new ContinuousFeatureBinariser(thresholds(idx))
    }
  }

  /**
   * Samples thresholds that will be used in continuous feature binarisation.
   */
  def sampleThresholds(data: RDD[LabeledPoint], featureIndices: List[Int]): Map[Int, Double] = {
    data.map { p =>
      val features = p.features.toArray
      val selected = featureIndices.map(features)

      val marked = selected.map(x => List((x, Random.nextFloat())))
      marked
    }.reduce{(list1, list2) =>
      (list1 zip list2).map{ case (el1, el2) => (el1 ++ el2).sortBy(_._2).take(2)}
    }.map(list => list.unzip._1.sum / 2).zip(featureIndices).map(_.swap).toMap
  }

  /**
   * Samples subsets that will be used in categorical feature binarisation.
   */
  def sampleSubsets(categoricalFeaturesInfo: Map[Int, Int]): Map[Int, BitSet] = {
    def randomBitSet(maxElem: Int): BitSet = {
      val bitsInLong = 64
      val longsNeeded = math.ceil(maxElem.toDouble / bitsInLong).toInt

      BitSet.fromBitMaskNoCopy(Array.fill(longsNeeded)(Random.nextLong()))
    }

    categoricalFeaturesInfo.mapValues(randomBitSet)
  }

  def sampleFeatureIndices(data: RDD[LabeledPoint], numFeatures: Int): List[Int] = {
    val numFeaturesInData = data.first().features.size
    sampleFeatureIndices(numFeaturesInData, numFeatures)
  }

  def sampleFeatureIndices(numFeaturesInData: Int, numFeatures: Int): List[Int] = {
    Random.shuffle((0 until numFeaturesInData).toList).take(numFeatures).sorted
  }

  def computeScores(aggregated: Array[((Double, Int), Long)], numDistinctPoints: Int, labels: Array[Double]) = {
    val labelsRev = labels.toList.zipWithIndex.toMap
    val numLabels = labels.length

    val objectsInLeafPerLabel = Array.fill[Long](numLabels, numDistinctPoints)(0)
    val objectsInLeaf = Array.fill[Long](numDistinctPoints)(0)
    val objectsPerLabel = Array.fill[Long](numLabels)(0)

    aggregated.foreach { case ((label, pointIdx), count) =>
      val labelIdx = labelsRev(label)
      objectsInLeafPerLabel(labelIdx)(pointIdx) += count
      objectsInLeaf(pointIdx) += count
      objectsPerLabel(labelIdx) += count
    }

    val numSamples = objectsPerLabel.sum

    val countOfZeros = objectsInLeafPerLabel.map(_.count(_ == 0)).sum
    val countOfMin = objectsInLeafPerLabel.flatMap(_.filter(_ > 0)).groupBy(identity).minBy(_._1)._2.length
    val epsilon =
      if (countOfZeros > 0) {
        countOfMin.toDouble / (countOfZeros * numSamples)
      } else {
        0.0
      }

    val scores = Array.tabulate[Double](numLabels, numDistinctPoints) { case (label, pointIdx) => log(
      (objectsInLeafPerLabel(label)(pointIdx) + epsilon)/(objectsInLeaf(pointIdx) + numLabels * epsilon)
        *
        (numSamples + numLabels * epsilon)/(objectsPerLabel(label) + epsilon)
    )}

    scores
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

  def train(input: RDD[LabeledPoint], numFeatures: Int, categoricalFeaturesInfo: Map[Int, Int], labels: Array[Double]): FernModel = {
    val featureIndices = sampleFeatureIndices(input, numFeatures)
    val binarisers = sampleBinarisers(input, featureIndices, categoricalFeaturesInfo)

    new Fern(Some(labels)).run(input, featureIndices, binarisers)
  }

  def trainAndAssess(input: RDD[LabeledPoint], numFeatures: Int, categoricalFeaturesInfo: Map[Int, Int], labels: Array[Double]): FernModelWithStats = {
    val featureIndices = sampleFeatureIndices(input, numFeatures)
    val binarisers = sampleBinarisers(input, featureIndices, categoricalFeaturesInfo)

    new Fern(Some(labels)).runAndAssess(input, featureIndices, binarisers)
  }

  def train(input: RDD[LabeledPoint], featureIndices: List[Int], labels: Array[Double]): FernModel =
    new Fern(Some(labels)).run(input, featureIndices, sampleBinarisers(input, featureIndices, Map.empty))

  def train(input: RDD[LabeledPoint], numFeatures: Int, labels: Array[Double], binarisers: List[FeatureBinariser]): FernModel =
    new Fern(Some(labels)).run(input, sampleFeatureIndices(input, numFeatures), binarisers)

  def train(input: RDD[LabeledPoint], featureIndices: List[Int], labels: Array[Double], binarisers: List[FeatureBinariser]): FernModel =
    new Fern(Some(labels)).run(input, featureIndices, binarisers)

  def train(input: RDD[LabeledPoint], numFeatures: Int, binarisers: List[FeatureBinariser]): FernModel =
    new Fern(None).run(input, sampleFeatureIndices(input, numFeatures), binarisers)

  def train(input: RDD[LabeledPoint], featureIndices: List[Int], binarisers: List[FeatureBinariser]): FernModel =
    new Fern(None).run(input, featureIndices, binarisers)
}
