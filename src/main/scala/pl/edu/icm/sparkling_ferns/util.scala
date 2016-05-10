package pl.edu.icm.sparkling_ferns

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * @author Mateusz Fedoryszak (mfedoryszak@gmail.com)
 */
object util {
  def arrayReduction[T : Manifest](f: (T, T) => T)(a1: Array[T], a2: Array[T]): Array[T] = {
    val minLen = math.min(a1.length, a2.length)
    Array.tabulate(minLen)(i => f(a1(i), a2(i)))
  }

  def mean[T](s: Traversable[T])(implicit n: Fractional[T]) = n.div(s.sum, n.fromInt(s.size))

  def extractLabels(data: RDD[LabeledPoint]) =
    data.map(p => p.label).distinct().collect()
}
