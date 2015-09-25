package pl.edu.icm.sparkling_ferns

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest.FunSuite

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class FernForestSuite extends FunSuite with LocalSparkContext with FixedRandomNumGenSeed {
  test("Simple fern forest") {
    val dataset = Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0))
    )

    val rdd = sc.parallelize(dataset)

    val model = FernForest.train(rdd, 3, 2, Map.empty)

    assert(model.predict(Vectors.dense(1.0, 1.0, 1.0)) == 1.0)
    assert(model.predict(Vectors.dense(-1.0, -1.0, -1.0)) == -1.0)
  }

  test("Simple fern forest with categorical features") {
    val dataset = Seq(
      LabeledPoint(1.0, Vectors.dense(0.0, 0.0, 0.0)),
      LabeledPoint(1.0, Vectors.dense(0.0, 0.0, 0.0)),
      LabeledPoint(1.0, Vectors.dense(2.0, 2.0, 2.0)),
      LabeledPoint(1.0, Vectors.dense(2.0, 2.0, 2.0)),
      LabeledPoint(-1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(1.0, 1.0, 1.0))
    )

    val rdd = sc.parallelize(dataset)

    val model = FernForest.train(rdd, 7, 2, Map(0 -> 3, 1 -> 3, 2 -> 3))

    assert(model.predict(Vectors.dense(0.0, 0.0, 0.0)) == 1.0)
    assert(model.predict(Vectors.dense(1.0, 1.0, 1.0)) == -1.0)
    assert(model.predict(Vectors.dense(2.0, 2.0, 2.0)) == 1.0)
  }
}
