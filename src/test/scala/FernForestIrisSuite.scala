import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest.FunSuite
import resource._

import scala.io.Source
import scala.util.Random

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
class FernForestIrisSuite extends FunSuite with LocalSparkContext {
  test("Simple integration test on Iris dataset") {
    val namedPoints = managed(Source.fromInputStream(getClass.getResourceAsStream("iris.csv"))).acquireAndGet {source =>
      val lines = source.getLines().filterNot(_.trim.isEmpty).toList
      val parts = lines.map(_.split(","))
      parts.map(p => (p(4), p.take(4).map(_.toDouble)))
    }

    val names = namedPoints.unzip._1.distinct.zipWithIndex.toMap

    val labeledPoints = namedPoints.map(r => LabeledPoint(names(r._1), Vectors.dense(r._2)))

    val rdd = sc.parallelize(labeledPoints)

    val markedPoints = rdd.map(p => (p, Random.nextDouble()))

    val training = markedPoints.filter(_._2 < 0.7).map(_._1)
    val testing = markedPoints.filter(_._2 >= 0.7).map(_._1)

    val modelAndStats = FernForest.trainAndAssess(training, 10, 3, Map.empty)
    val accuracy = testing.map(p =>
      if (modelAndStats.model.predict(p.features) == p.label) 1.0 else 0.0
    ).mean()

    println(s"Accuracy: $accuracy")
    assert(accuracy > 0.5)
  }
}
