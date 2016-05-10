package pl.edu.icm.sparkling_ferns

import org.scalatest.{BeforeAndAfterEach, Suite}

import scala.util.Random

/**
 * @author Mateusz Fedoryszak (mfedoryszak@gmail.com)
 */
trait FixedRandomNumGenSeed extends BeforeAndAfterEach { self: Suite =>
  override def beforeEach() {
    Random.setSeed(0)
    super.beforeEach()
  }
}