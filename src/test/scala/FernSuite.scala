import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest._

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class FernSuite extends FunSuite with LocalSparkContext {
  test("Simple Fern") {
    val dataset = Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0))
    )

    val rdd = sc.parallelize(dataset)

    val model = new Fern().run(rdd, 1)

    assert(model.predict(Vectors.dense(1.0, 1.0, 1.0)) == 1.0)
    assert(model.predict(Vectors.dense(-1.0, -1.0, -1.0)) == -1.0)
  }

  test("Manual feature selection fern") {
    val dataset = Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, 1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, 1.0, 1.0))
    )

    val rdd = sc.parallelize(dataset)

    val model = new Fern().run(rdd, List(0))

    assert(model.predict(Vectors.dense(1.0, 1.0, 1.0)) == 1.0)
    assert(model.predict(Vectors.dense(-1.0, -1.0, -1.0)) == -1.0)

    assert(model.predict(Vectors.dense(1.0, -1.0, -1.0)) == 1.0)
    assert(model.predict(Vectors.dense(-1.0, 1.0, 1.0)) == -1.0)
  }
}
