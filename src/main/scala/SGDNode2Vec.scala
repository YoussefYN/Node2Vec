
import java.io.File

import breeze.linalg.{Axis, DenseMatrix, DenseVector, csvread, sum}
import breeze.numerics.{log, sigmoid}
import org.apache.spark.rdd.RDD

import scala.util.Random

@SerialVersionUID(15L)
class SGDNode2Vec(embSize: Int, nodes: Int, loadParams: Boolean = false) extends Serializable {
  val rand = new Random()

  val embeddingIn: DenseMatrix[Double] =
    if (loadParams) csvread(new File("emb_in.csv"), separator = ',')
    else new DenseMatrix(rows = embSize, cols = nodes, Array.fill(nodes * embSize)(rand.nextDouble()))

  val embeddingOut: DenseMatrix[Double] =
    if (loadParams) csvread(new File("emb_out.csv"), separator = ',')
    else new DenseMatrix(rows = embSize, cols = nodes, Array.fill(nodes * embSize)(rand.nextDouble()))

  def fit(batches: Array[RDD[(Int, Int)]], learningRate: Double,
          epochs: Int, negativeSamples: Int, batchSize: Double,
          testData: RDD[(Int, Int)])
  : (DenseMatrix[Double], DenseMatrix[Double]) = {

    for (epoch <- 1 to epochs) {

      var error = 0.0
      for (batch <- batches) {
        val gradients = batch.map(edge => estimateEdgeGradients(edge._1, edge._2, negativeSamples,
          embeddingIn, embeddingOut))
        val inGrads = gradients.map(x => x._1).reduceByKey(_ + _)
        val outGrads = gradients.map(x => x._2).reduceByKey(_ + _)
        inGrads.collect().foreach(x => {
          embeddingIn(::, x._1) += -learningRate * x._2 / batchSize
        })
        outGrads.collect().foreach(x => {
          embeddingOut(::, x._1) += -learningRate * x._2 / batchSize
        })
        error += getBatchError(batch, negativeSamples)
      }
      val testError = getTestError(testData)
      println("Epoch", epoch, "Train error:", error, "Test error:", testError)
    }
    (embeddingIn, embeddingOut)
  }

  def getTestError(testData: RDD[(Int, Int)]): Double = {
    testData.map(edge => estimateTestError(edge._1, edge._2, embeddingIn, embeddingOut)).reduce(_ + _) / testData.count()
  }

  def getBatchError(batch: RDD[(Int, Int)], negativeSamples: Int): Double = {
    batch.map(edge => estimateEdgeError(edge._1, edge._2, negativeSamples,
      embeddingIn, embeddingOut)).reduce(_ + _) / batch.count()
  }

  def estimateEdgeGradients(source: Int, destination: Int, negatives: Int,
                            embInBroadcast: DenseMatrix[Double], embOutBroadcast: DenseMatrix[Double]):
  ((Int, DenseVector[Double]), (Int, DenseVector[Double])) = {

    val in = embInBroadcast(::, source)
    val out = embOutBroadcast(::, destination)
    val negSamples = embOutBroadcast(::, IndexedSeq.fill(negatives)(rand.nextInt(nodes))).toDenseMatrix

    val sigmoids = sigmoid.apply(negSamples.t * in)
    val temp = negSamples.mapPairs({
      case ((_, col), value) => value * sigmoids(col)
    })
    val inGrad = out * -sigmoid.apply(-in.dot(out)) + sum(temp, Axis._1)
    val outGrad = in * -sigmoid.apply(-in.dot(out))

    ((source, inGrad), (destination, outGrad))
  }

  def estimateTestError(source: Int, destination: Int,
                        embInBroadcast: DenseMatrix[Double], embOutBroadcast: DenseMatrix[Double]):
  Double = {

    val in = embInBroadcast(::, source)
    val out = embOutBroadcast(::, destination)

    -log(sigmoid.apply(in.dot(out)))
  }

  def estimateEdgeError(source: Int, destination: Int, negatives: Int,
                        embInBroadcast: DenseMatrix[Double], embOutBroadcast: DenseMatrix[Double]):
  Double = {

    val in = embInBroadcast(::, source)
    val out = embOutBroadcast(::, destination)
    val negSamples = embOutBroadcast(::, IndexedSeq.fill(negatives)(rand.nextInt(nodes))).toDenseMatrix

    val sigmoids = sigmoid.apply(negSamples.t * in)
    val negativeTerm = sum(log(1.0 - sigmoids))
    -log(sigmoid.apply(in.dot(out))) - negativeTerm
  }
}
