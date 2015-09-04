import org.scalatest.{BeforeAndAfterEach, Suite}

import scala.util.Random

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
trait FixedRandomNumGenSeed extends BeforeAndAfterEach { self: Suite =>
  override def beforeEach() {
    Random.setSeed(0)
    super.beforeEach()
  }
}