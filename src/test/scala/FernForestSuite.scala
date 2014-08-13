import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest.FunSuite

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class FernForestSuite extends FunSuite with LocalSparkContext {
  test("Simple fern forest") {
    val dataset = Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0))
    )

    val rdd = sc.parallelize(dataset)

    val model = FernForest.train(rdd, 3, 2)

    assert(model.predict(Vectors.dense(1.0, 1.0, 1.0)) == 1.0)
    assert(model.predict(Vectors.dense(-1.0, -1.0, -1.0)) == -1.0)
  }
}
