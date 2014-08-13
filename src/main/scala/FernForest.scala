import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class FernForestModel(private val ferns: List[FernModel]) extends ClassificationModel with Serializable {
  override def predict(testData: RDD[Vector]): RDD[Double] = testData.map(predict)

  override def predict(testData: Vector): Double = {
    val scores = ferns.map(_.scores(testData))
    val scoreSums = scores.reduce[Array[Double]]{case (a1, a2) => (a1 zip a2) map (x => x._1 + x._2)}
    val labels = ferns.head.labels
    val labelIdx = (0 until labels.length) maxBy scoreSums

    labels(labelIdx)
  }
}

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class FernForest {
  def run(data: RDD[LabeledPoint], numFerns: Int, numFeatures: Int): FernForestModel = {
    new FernForestModel(List.fill(numFerns)(Fern.train(data, numFeatures)))
  }

  def run(data: RDD[LabeledPoint], featureIndices: List[List[Int]]): FernForestModel = {
    new FernForestModel(featureIndices.map(Fern.train(data, _)))
  }
}

object FernForest {
  def train(input: RDD[LabeledPoint], numFerns: Int, numFeatures: Int): FernForestModel = {
    new FernForest().run(input, numFerns, numFeatures)
  }


  def train(input: RDD[LabeledPoint], featureIndices: List[List[Int]]): FernForestModel = {
    new FernForest().run(input, featureIndices)
  }
}
