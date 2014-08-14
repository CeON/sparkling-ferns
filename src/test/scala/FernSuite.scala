import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest._

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class FernSuite extends FunSuite with LocalSparkContext {
  test("Simple fern") {
    val dataset = Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0))
    )

    val rdd = sc.parallelize(dataset)

    val model = Fern.train(rdd, 1, List.fill(3)(0.0))

    assert(model.predict(Vectors.dense(1.0, 1.0, 1.0)) == 1.0)
    assert(model.predict(Vectors.dense(-1.0, -1.0, -1.0)) == -1.0)
  }


  test("All features fern") {
    val dataset = Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0))
    )

    val rdd = sc.parallelize(dataset)

    val model = Fern.train(rdd, 3, List.fill(3)(0.0))

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

    val model = Fern.train(rdd, List(0), List.fill(3)(0.0))

    assert(model.predict(Vectors.dense(1.0, 1.0, 1.0)) == 1.0)
    assert(model.predict(Vectors.dense(-1.0, -1.0, -1.0)) == -1.0)

    assert(model.predict(Vectors.dense(1.0, -1.0, -1.0)) == 1.0)
    assert(model.predict(Vectors.dense(-1.0, 1.0, 1.0)) == -1.0)
  }
}
