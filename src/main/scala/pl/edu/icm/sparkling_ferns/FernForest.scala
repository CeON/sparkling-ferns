package pl.edu.icm.sparkling_ferns

import breeze.stats.distributions.Poisson
import org.apache.spark.SparkContext._
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
    runAndAssess(data, numFerns, numFeatures, categoricalFeaturesInfo).model
    //val labels = util.extractLabels(data)
    //new FernForestModel(List.fill(numFerns)(Fern.train(data, numFeatures, categoricalFeaturesInfo, labels)))
  }

  def runAndAssess(data: RDD[LabeledPoint], numFerns: Int, numFeatures: Int, categoricalFeaturesInfo: Map[Int, Int]): FernForestModelWithStats = {
    val labels = util.extractLabels(data)

    val numFeaturesInData = data.take(1).head.features.size

    val withMultipliers = data.map(x => (x, Array.fill(numFerns)(Poisson.distribution(1.0).draw())))

    val featureIndicesPerFern = Array.fill(numFerns)(Fern.sampleFeatureIndices(numFeaturesInData, numFeatures))

    val binarisersPerFern = Array.tabulate(numFerns)(i =>
      Fern.sampleBinarisers(
        withMultipliers.flatMap{case (point, muls) => List.fill(muls(i))(point)},
        featureIndicesPerFern(i), categoricalFeaturesInfo))

    val fernBuilders = (0 until numFerns).map{i =>
      new FernBuilder(featureIndicesPerFern(i), binarisersPerFern(i))
    }

    val counts = withMultipliers.flatMap { case (point, muls) =>
      (0 until numFerns).map { i =>
        ((i, point.label, fernBuilders(i).toIndex(point.features)), muls(i).toLong)
      }
    }.reduceByKey(_ + _).collect()

    val countsPerFern = counts.groupBy(_._1._1).mapValues(_.map{ case ((_, label, idx), count) => (label, idx) -> count})

    val ferns = (0 until numFerns).toList.map { i => fernBuilders(i).build(countsPerFern(i), labels)}

    val model = new FernForestModel(ferns)

    val confusionMatrix = withMultipliers.flatMap{ case (point, muls) =>
      val fernIndices = muls.toList.zipWithIndex.filter(_._1 == 0).map(_._2)
      fernIndices.map(ferns).map(fern => ((point.label, fern.predict(point.features)), 1l))
    }.reduceByKey(_ + _).collect().toList

    val modelsWithStats = List.fill(numFerns)(Fern.trainAndAssess(data, numFeatures, categoricalFeaturesInfo, labels))

    val featureImportance = Nil //modelsWithStats.flatMap(_.featureImportance).groupBy(_._1).map{case (idx, list) => (idx, util.mean(list.unzip._2))}.toList

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
