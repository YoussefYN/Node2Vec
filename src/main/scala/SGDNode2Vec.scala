
import breeze.linalg.DenseMatrix
import org.apache.spark.rdd.RDD


class SGDNode2Vec(embSize: Int, nodes: Int) {
  val embeddingIn = new DenseMatrix[Double](rows = nodes, cols = embSize)
  val embeddingOut = new DenseMatrix[Double](rows = nodes, cols = embSize)

  def fit(batches: Array[RDD[(Int, Int)]], learningRate: Double, epochs: Int, negativeSamples: Int)
  : (DenseMatrix[Double], DenseMatrix[Double]) = {
    (embeddingIn, embeddingOut)
  }
}
