/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
object util {
  def arrayReduction[T : Manifest](f: (T, T) => T)(a1: Array[T], a2: Array[T]): Array[T] = {
    val minLen = math.min(a1.length, a2.length)
    Array.tabulate(minLen)(i => f(a1(i), a2(i)))
  }
}
