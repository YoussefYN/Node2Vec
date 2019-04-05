import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}

class EdgesHelper(session: SparkSession, BATCH_SIZE: Int) {

  /* Return RDD of tuple (int, int) for each edge in the input file. */
  def loadEdges(path: String): RDD[(Int, Int)] = {
    session.read.format("csv")
      // header: source_node, destination_node
      // here we read the data from CSV and export it as RDD[(Int, Int)],
      // i.e. as RDD of edges
      .option("header", "true")
      // State that the header is present in the file
      .schema(StructType(Array(
      StructField("source_node", IntegerType, false),
      StructField("destination_node", IntegerType, false)
    )))
      // Define schema of the input data
      .load(path)
      // Read the file as DataFrame
      .rdd.map(row => (row.getAs[Int](0), row.getAs[Int](1)))
    // Interpret DF as RDD
  }

  /* Return Batches of RDDs*/
  def get_edges_batches(edgesRdd: RDD[(Int, Int)], batchSize: Int): Array[RDD[(Int, Int)]] = {
    val edges_count: Double = edgesRdd.count()

    // calculate number of batches
    var partitions = (edges_count / BATCH_SIZE).toInt
    if (edges_count % BATCH_SIZE != 0) {
      partitions += 1
    }

    // creating array for weight split
    val weights = new Array[Double](partitions)
    // weight of one batch in relation to whole dataset
    val weight: Double = BATCH_SIZE / edges_count

    var pos = 0
    var index = 0
    // filling weights array
    while (pos + BATCH_SIZE < edges_count) {
      weights(index) = weight
      pos += BATCH_SIZE
      index += 1
    }
    // the least batch can weight less
    if (pos < edges_count) {
      weights(index) = 1 - weights.sum
    }

    edgesRdd.randomSplit(weights) // splitting
  }
}
