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
    val epoch = 20
    val learningRate = 2.7
    val embeddingSize = 50
    val negativeSamples = 20

    val helper = new EdgesHelper(session, batchSize)
    val trainData = helper.loadEdges(args(0))

    val nodes = trainData.map(x => math.max(x._1, x._2)).reduce(math.max) + 1
    val batches = helper.get_edges_batches(trainData, batchSize)
    val node2Vec = new SGDNode2Vec(embeddingSize, nodes)
    val (embIn, embOut) = node2Vec.fit(batches, learningRate, epoch, negativeSamples, batchSize)
    csvwrite(new File("emb_in.csv"), embIn, separator = ',')
    csvwrite(new File("emb_out.csv"), embOut, separator = ',')

  }
}
