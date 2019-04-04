import org.apache.spark.{SparkConf, SparkContext}

object Main {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Node2Vec").setMaster("local")
    val sc = new SparkContext(conf)
    val helper = new EdgesHelper(sc)

    val trainData = helper.loadEdges(args(0))
    val batchSize = 10000
    val epoch = 20
    val learningRate = 1.0
    val embeddingSize = 50
    val negativeSamples = 20
    val nodes = 40334
    val batches = helper.get_edges_batches(trainData, batchSize)
    val node2Vec = new SGDNode2Vec(embeddingSize, nodes)
    node2Vec.fit(batches, learningRate, epoch, negativeSamples)
  }
}
