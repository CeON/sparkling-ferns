package pl.edu.icm.sparkling_ferns

import scala.collection.BitSet

trait FeatureBinariser extends Serializable {
  def apply(v: Double): Boolean
}

class ContinuousFeatureBinariser(private val threshold: Double) extends FeatureBinariser {
  override def apply(v: Double): Boolean = v > threshold
}

class CategoricalFeatureBinariser(private val selected: BitSet) extends FeatureBinariser {
  override def apply(v: Double): Boolean = selected.contains(v.toInt)
}
