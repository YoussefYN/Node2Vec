import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class EdgesHelper(sc: SparkContext) {

  /* Return RDD of tuple (int, int) for each edge in the input file. */
  def loadEdges(path: String): RDD[(Int, Int)] = {
    // TODO
    return sc.emptyRDD
  }

  /* Return Batches of RDDs*/
  def get_edges_batches(edgesRdd: RDD[(Int, Int)], batchSize: Int): Array[RDD[(Int, Int)]] = {
    // TODO
    return Array()
  }
}
