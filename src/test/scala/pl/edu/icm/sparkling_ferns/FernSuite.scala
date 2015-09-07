package pl.edu.icm.sparkling_ferns

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest._

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class FernSuite extends FunSuite with LocalSparkContext with FixedRandomNumGenSeed {
  test("Simple fern") {
    val dataset = Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0))
    )

    val rdd = sc.parallelize(dataset)

    val model = Fern.train(rdd, 1, List.fill(3)(0.0).map(new ContinuousFeatureBinariser(_)))

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

    val model = Fern.train(rdd, 3, List.fill(3)(0.0).map(new ContinuousFeatureBinariser(_)))

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

    val model = Fern.train(rdd, List(0), List.fill(3)(0.0).map(new ContinuousFeatureBinariser(_)))

    assert(model.predict(Vectors.dense(1.0, 1.0, 1.0)) == 1.0)
    assert(model.predict(Vectors.dense(-1.0, -1.0, -1.0)) == -1.0)

    assert(model.predict(Vectors.dense(1.0, -1.0, -1.0)) == 1.0)
    assert(model.predict(Vectors.dense(-1.0, 1.0, 1.0)) == -1.0)
  }

  test("Confusion matrix") {
    val dataset = Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0))
    )

    val rdd = sc.parallelize(dataset)

    val model = Fern.train(rdd, List(0), List.fill(3)(0.0).map(new ContinuousFeatureBinariser(_)))

    val validationSet = Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0, -1.0))
    )

    val validationRdd = sc.parallelize(validationSet)

    val confusionMatrix = model.confusionMatrix(validationRdd).toMap.withDefault(_ => 0)

    assert(confusionMatrix((1.0, 1.0)) == 3)
    assert(confusionMatrix((1.0, -1.0)) == 1)
    assert(confusionMatrix((-1.0, -1.0)) == 2)
    assert(confusionMatrix((-1.0, 1.0)) == 0)
  }

  test("Feature importance") {
    val dataset = Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, -1.0))
    )

    val rdd = sc.parallelize(dataset)

    val model = Fern.train(rdd, List(0, 1), List.fill(3)(0.0).map(new ContinuousFeatureBinariser(_)))

    val validationSet = Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, 1.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, 1.0))
    )

    val validationRdd = sc.parallelize(validationSet)

    val importance = model.featureImportance(validationRdd).toMap.withDefault(_ => 0.0)

    assert(importance(0) > importance(1))
  }
}
