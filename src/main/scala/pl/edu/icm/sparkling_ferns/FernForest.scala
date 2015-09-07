package pl.edu.icm.sparkling_ferns

import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class FernForestModel(private val ferns: List[FernModel]) extends ClassificationModel with Serializable {
  override def predict(testData: RDD[Vector]): RDD[Double] = testData.map(predict)

  override def predict(testData: Vector): Double = {
    val scores = ferns.map(_.scores(testData))
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
    val labels = util.extractLabels(data)
    new FernForestModel(List.fill(numFerns)(Fern.train(data, numFeatures, categoricalFeaturesInfo, labels)))
  }

  def runAndAssess(data: RDD[LabeledPoint], numFerns: Int, numFeatures: Int, categoricalFeaturesInfo: Map[Int, Int]): FernForestModelWithStats = {
    val labels = util.extractLabels(data)
    val modelsWithStats = List.fill(numFerns)(Fern.trainAndAssess(data, numFeatures, categoricalFeaturesInfo, labels))

    val featureImportance = modelsWithStats.flatMap(_.featureImportance).groupBy(_._1).map{case (idx, list) => (idx, util.mean(list.unzip._2))}.toList
    val confusionMatrix = modelsWithStats.flatMap(_.oobConfusionMatrix).groupBy(_._1).map{case (cell, list) => (cell, list.unzip._2.sum)}.toList

    val model = new FernForestModel(modelsWithStats.map(_.model))

    FernForestModelWithStats(model, confusionMatrix, featureImportance)
  }

  def run(data: RDD[LabeledPoint], featureIndices: List[List[Int]]): FernForestModel = {
    val labels = util.extractLabels(data)
    new FernForestModel(featureIndices.map(Fern.train(data, _, labels)))
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

  def train(input: RDD[LabeledPoint], featureIndices: List[List[Int]]): FernForestModel =
    new FernForest().run(input, featureIndices)
}
