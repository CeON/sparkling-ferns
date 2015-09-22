package pl.edu.icm.sparkling_ferns

import breeze.stats.distributions.Poisson
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class FernForestModel(val ferns: Array[FernModel]) extends ClassificationModel with Serializable {
  override def predict(testData: RDD[Vector]): RDD[Double] = testData.map(predict)

  override def predict(testData: Vector): Double = {
    predictSubset(testData, 0 until ferns.length)
  }

  /**
   * Make a prediction using only a subsed of ferns specified by fernIdcs parameter.
   */
  def predictSubset(testData: Vector, fernIdcs: TraversableOnce[Int]): Double = {
    val scores = fernIdcs.map(ferns).map(_.scores(testData))
    val scoreSums = scores.reduce(util.arrayReduction[Double](_ + _))
    val labels = ferns.head.labels
    val labelIdx = (0 until labels.length) maxBy scoreSums

    labels(labelIdx)
  }
}

case class FernForestModelWithStats(model: FernForestModel, oobConfusionMatrix: List[((Double, Double), Long)], featureImportance: List[(Int, Double)])

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class FernForest {
  def run(data: RDD[LabeledPoint], numFerns: Int, numFeatures: Int, categoricalFeaturesInfo: Map[Int, Int]): FernForestModel = {
    val withMultipliers = data.map(x => (x, Array.fill(numFerns)(Poisson.distribution(1.0).draw())))

    runWithMultipliers(withMultipliers, numFerns, numFeatures, categoricalFeaturesInfo)
  }

  def runAndAssess(data: RDD[LabeledPoint], numFerns: Int, numFeatures: Int, categoricalFeaturesInfo: Map[Int, Int]): FernForestModelWithStats = {
    val withMultipliers = data.map(x => (x, Array.fill(numFerns)(Poisson.distribution(1.0).draw())))

    val model = runWithMultipliers(withMultipliers, numFerns, numFeatures, categoricalFeaturesInfo)

    val confusionMatrix = withMultipliers.filter(_._2.contains(0)).map{ case (point, muls) =>
      val fernIndices = muls.toList.zipWithIndex.filter(_._1 == 0).map(_._2)
      ((point.label, model.predictSubset(point.features, fernIndices)), 1l)
    }.reduceByKey(_ + _).collect().toList


    //TODO: constant number of passes implementation
    val featureImportance = model.ferns.zipWithIndex
      .flatMap{ case (fern, i) => fern.featureImportance(withMultipliers.filter(_._2(i) == 0).map(_._1)) }
      .groupBy(_._1).mapValues(_.unzip._2).mapValues(util.mean(_)).toList

    FernForestModelWithStats(model, confusionMatrix, featureImportance)
  }

  def runWithMultipliers(withMultipliers: RDD[(LabeledPoint, Array[Int])], numFerns: Int, numFeatures: Int, categoricalFeaturesInfo: Map[Int, Int]): FernForestModel = {
    val metadata = DatasetMetadata.fromData(withMultipliers.map(_._1))

    val featureIndicesPerFern = Array.fill(numFerns)(Fern.sampleFeatureIndices(metadata.numFeatures, numFeatures))

    val thresholds = withMultipliers.flatMap { case(point, muls) =>
      val features = point.features.toArray
      for {
        fernIdx <- 0 until numFerns
        featureIdx <- featureIndicesPerFern(fernIdx) if !categoricalFeaturesInfo.contains(featureIdx)
        _ <- 0 until muls(fernIdx)
      } yield ((fernIdx, featureIdx), List((Random.nextFloat(), features(featureIdx))))
    }.reduceByKey((list1, list2) => (list1 ++ list2).sortBy(_._1).take(2))
      .mapValues(_.unzip._2).mapValues(list => list.sum / list.size).collect()

    val thresholdsPerFern = thresholds.groupBy(_._1._1).mapValues(
      _.map{case ((fernIdx, featureIdx), threshold) => (featureIdx, threshold)}.toMap)

    val fernBuilders = (0 until numFerns).map{i =>
      new FernBuilder(featureIndicesPerFern(i), thresholdsPerFern.getOrElse(i, Map.empty), categoricalFeaturesInfo)
    }

    val counts = withMultipliers.flatMap { case (point, muls) =>
      (0 until numFerns).map { i =>
        ((i, point.label, fernBuilders(i).toCombinationIndex(point.features)), muls(i).toLong)
      }
    }.reduceByKey(_ + _).collect()

    val countsPerFern = counts.groupBy(_._1._1).mapValues(_.map{ case ((_, label, idx), count) => (label, idx) -> count})

    val ferns = (0 until numFerns).map{ i => fernBuilders(i).build(countsPerFern(i), metadata.labels)}.toArray

    new FernForestModel(ferns)
  }
}

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
object FernForest {
  def train(input: RDD[LabeledPoint], numFerns: Int, numFeatures: Int, categoricalFeaturesInfo: Map[Int, Int]): FernForestModel =
    new FernForest().run(input, numFerns, numFeatures, categoricalFeaturesInfo)

  def trainAndAssess(input: RDD[LabeledPoint], numFerns: Int, numFeatures: Int, categoricalFeaturesInfo: Map[Int, Int]): FernForestModelWithStats =
    new FernForest().runAndAssess(input, numFerns, numFeatures, categoricalFeaturesInfo)
}
