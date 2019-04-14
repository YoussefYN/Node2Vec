import java.io.File

import breeze.linalg.csvwrite
import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {

    val session = SparkSession
      .builder()
      .appName("Node2Vec")
      .master("local")
      .getOrCreate()

    val sc = session.sparkContext
    sc.setLogLevel("OFF")

    val batchSize = 10000
    val epoch = 10
    val learningRate = 27
    val embeddingSize = 35
    val negativeSamples = 20
    val nodes = 40334

    val helper = new EdgesHelper(session, batchSize)
    val trainData = helper.loadEdges(args(0))
    val batches = helper.get_edges_batches(trainData, batchSize)

    val testData = helper.loadEdges(args(1))

    val node2Vec = new SGDNode2Vec(embeddingSize, nodes)
    val (embIn, embOut) = node2Vec.fit(batches, learningRate, epoch, negativeSamples, batchSize, testData)
    csvwrite(new File("emb_in.csv"), embIn, separator = ',')
    csvwrite(new File("emb_out.csv"), embOut, separator = ',')

  }
}
