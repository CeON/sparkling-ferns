package pl.edu.icm.sparkling_ferns

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
case class DatasetMetadata(labels: Array[Double], numFeatures: Int)

object DatasetMetadata {
  def fromData(data: RDD[LabeledPoint]): DatasetMetadata = {
    val labels = util.extractLabels(data)

    val numFeaturesInData = data.take(1).head.features.size

    DatasetMetadata(labels, numFeaturesInData)
  }
}
