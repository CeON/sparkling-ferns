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
class FernForestIntegrationSuite extends FunSuite with LocalSparkContext {
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

  test("Simple integration test on Car dataset") {
    def generateMap(names: List[String]) =
      names.zipWithIndex.toMap.mapValues(_.toDouble)

    val classMap = generateMap(List("unacc", "acc", "good", "vgood"))
    val buyingMap = generateMap(List("vhigh", "high", "med", "low"))
    val maintMap = generateMap(List("vhigh", "high", "med", "low"))
    val doorsMap = generateMap(List("2", "3", "4", "5more"))
    val personsMap = generateMap(List("2", "4", "more"))
    val lugBootMap = generateMap(List("small", "med", "big"))
    val safetyMap = generateMap(List("low", "med", "high"))

    val labeledPoints = managed(Source.fromInputStream(getClass.getResourceAsStream("car.csv"))).acquireAndGet {source =>
      val lines = source.getLines().filterNot(_.trim.isEmpty).toList
      val parts = lines.map(_.split(","))
      parts.map(p => LabeledPoint(classMap(p(6)), Vectors.dense(buyingMap(p(0)), maintMap(p(1)), doorsMap(p(2)),
        personsMap(p(3)), lugBootMap(p(4)), safetyMap(p(5)))))
    }

    val balancedLabeledPoints = labeledPoints.groupBy(_.label).mapValues(list => Random.shuffle(list).take(65)).values.flatten.toList

    val rdd = sc.parallelize(balancedLabeledPoints)

    val markedPoints = rdd.map(p => (p, Random.nextDouble()))

    val training = markedPoints.filter(_._2 < 0.7).map(_._1)
    val testing = markedPoints.filter(_._2 >= 0.7).map(_._1)

    val modelAndStats = FernForest.trainAndAssess(training, 20, 4, Map(0->buyingMap.size, 1->maintMap.size, 2->doorsMap.size, 3->personsMap.size, 4->lugBootMap.size, 5->safetyMap.size))
    val accuracy = testing.map(p =>
      if (modelAndStats.model.predict(p.features) == p.label) 1.0 else 0.0
    ).mean()

    println(s"Accuracy: $accuracy")
    assert(accuracy > 0.5)
  }
}
