package kmeans

import scala.annotation.tailrec
import scala.collection._
import scala.util.Random
import org.scalameter._
import common._

class KMeans {

  def generatePoints(k: Int, num: Int): Seq[Point] = {
    val randx = new Random(1)
    val randy = new Random(3)
    val randz = new Random(5)
    (0 until num)
      .map({ i =>
        val x = ((i + 1) % k) * 1.0 / k + randx.nextDouble() * 0.5
        val y = ((i + 5) % k) * 1.0 / k + randy.nextDouble() * 0.5
        val z = ((i + 7) % k) * 1.0 / k + randz.nextDouble() * 0.5
        new Point(x, y, z)
      })
      .to[mutable.ArrayBuffer]
  }

  /**
    * Randomly choose k points
    *
    * @param k
    * @param points
    * @return
    */
  def initializeMeans(k: Int, points: Seq[Point]): Seq[Point] = {
    val rand = new Random(7)
    (0 until k)
      .map(_ => points(rand.nextInt(points.length)))
      .to[mutable.ArrayBuffer]
  }

  /**
    * Find the closest point to p point in means set
    * @param p
    * @param means
    * @return
    */
  def findClosest(p: Point, means: GenSeq[Point]): Point = {
    assert(means.nonEmpty)
    var minDistance = p.squareDistance(means.head)
    var closest = means.head
    var i = 1
    while (i < means.length) {
      val distance = p.squareDistance(means(i))
      if (distance < minDistance) {
        minDistance = distance
        closest = means(i)
      }
      i += 1
    }
    closest
  }

  /**
    * Returns a generic map collection, which maps each mean to the sequence of
    * points in the corresponding cluster.
    *
    * @param points
    * @param means
    * @return
    */
  def classify(points: GenSeq[Point], means: GenSeq[Point]): GenMap[Point, GenSeq[Point]] = {
    val resp: GenMap[Point, GenSeq[Point]] = points.groupBy(point => findClosest(point, means))
    val respKeys: GenSeq[Point] = resp.keySet.to[GenSeq]
    val missingKeys: GenSeq[Point] = means.diff(respKeys)
    resp ++ missingKeys.map(mean => mean -> GenSeq.empty)
  }

  /**
    * Find the new mean point on points set.
    *
    * @param oldMean
    * @param points
    * @return
    */
  def findAverage(oldMean: Point, points: GenSeq[Point]): Point =
    if (points.isEmpty) oldMean
    else {
      var x = 0.0
      var y = 0.0
      var z = 0.0
      points.seq.foreach { p =>
        x += p.x
        y += p.y
        z += p.z
      }
      new Point(x / points.length, y / points.length, z / points.length)
    }

  /**
    * Takes the map of classified points produced in the previous step,
    * and the sequence of previous means.
    * The method returns the new sequence of means.
    *
    * @param classified
    * @param oldMeans
    * @return
    */
  def update(classified: GenMap[Point, GenSeq[Point]], oldMeans: GenSeq[Point]): GenSeq[Point] = {
    var resp = GenSeq.empty[Point]
    oldMeans.map(oldMean => findAverage(oldMean, classified.getOrElse(oldMean, GenSeq.empty)))
  }

  /**
    * The algorithm converged iff the square distance between the old and the new mean is less
    * than or equal to eta, for all means.
    *
    * @param eta
    * @param oldMeans
    * @param newMeans
    * @return
    */
  def converged(eta: Double)(oldMeans: GenSeq[Point], newMeans: GenSeq[Point]): Boolean = {
    oldMeans
      .zip(newMeans)
      .map {
        case (oldMean, newMean) =>
          oldMean.squareDistance(newMean) <= eta
      }
      .forall(b => b)
  }

  @tailrec
  final def kMeans(points: GenSeq[Point], means: GenSeq[Point], eta: Double): GenSeq[Point] = {
    val classified = classify(points, means)
    val newMeans = update(classified, means)
    if (!converged(eta)(means, newMeans)) kMeans(points, newMeans, eta) else newMeans // your implementation need to be tail recursive
  }
}

/** Describes one point in three-dimensional space.
  *
  *  Note: deliberately uses reference equality.
  */
class Point(val x: Double, val y: Double, val z: Double) {
  private def square(v: Double): Double = v * v
  def squareDistance(that: Point): Double = {
    square(that.x - x) + square(that.y - y) + square(that.z - z)
  }
  private def round(v: Double): Double = (v * 100).toInt / 100.0
  override def toString = s"(${round(x)}, ${round(y)}, ${round(z)})"
}

object KMeansRunner {

  val standardConfig = config(
    Key.exec.minWarmupRuns -> 20,
    Key.exec.maxWarmupRuns -> 40,
    Key.exec.benchRuns -> 25,
    Key.verbose -> true
  ) withWarmer (new Warmer.Default)

  def main(args: Array[String]) {
    val kMeans = new KMeans()

    val numPoints = 500000
    val eta = 0.01
    val k = 32
    val points = kMeans.generatePoints(k, numPoints)
    val means = kMeans.initializeMeans(k, points)

    val seqtime = standardConfig measure {
      kMeans.kMeans(points, means, eta)
    }
    println(s"sequential time: $seqtime ms")

    val partime = standardConfig measure {
      val parPoints = points.par
      val parMeans = means.par
      kMeans.kMeans(parPoints, parMeans, eta)
    }
    println(s"parallel time: $partime ms")
    println(s"speedup: ${seqtime / partime}")
  }

}
