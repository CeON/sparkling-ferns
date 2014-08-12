import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

/**
 * @author Mateusz Fedoryszak (m.fedoryszak@icm.edu.pl)
 */
object SimpleApp {
  def main(args: Array[String]) {
    val logFile = "C:\\Users\\matfed\\Desktop\\krowy\\basicStats.r" // Should be some file on your system
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local")
    val sc = new SparkContext(conf)
    val logData = sc.textFile(logFile, 2).cache()
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
  }
}