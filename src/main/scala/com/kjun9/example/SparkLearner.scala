package com.kjun9.example

import org.apache.spark.sql.SparkSession
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.{
  ElementWiseVertex,
}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.api.RDDTrainingApproach
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.{MultiDataSet, api}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.util.Random

object SparkLearner {

  val spark = SparkSession.builder().master("local[*]").getOrCreate

  def main(args: Array[String]): Unit = {

    val batchSize = 5
    val featSize = 10
    val labelSize = 2
    val numBatches = 1000
    val numEpochs = 5
    val random = new Random(0)

    // dummy training data with 2 input feature matrices
    val trainingData: Seq[api.MultiDataSet] = (0 until numBatches).map { feat =>
      val features =
        Array(
          Nd4j.create(Array.fill(batchSize * featSize)(random.nextDouble),
                      Array(batchSize, featSize)),
          Nd4j.create(Array.fill(batchSize * featSize)(random.nextDouble),
                      Array(batchSize, featSize))
        )
      val labels =
        Array(
          Nd4j.create(Array.fill(batchSize * labelSize)(random.nextDouble),
                      Array(batchSize, labelSize)))
      new MultiDataSet(features, labels)
    }
    val trainingRdd = spark.sparkContext.parallelize(trainingData)

    // simple model
    val modelConf = new NeuralNetConfiguration.Builder()
      .updater(new Adam(0.01))
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .biasInit(0)
      .graphBuilder()
      .addInputs("input1", "input2")
      .addVertex(
        "avg",
        new ElementWiseVertex(ElementWiseVertex.Op.Average),
        "input1",
        "input2"
      )
      .addLayer(
        "dense",
        new DenseLayer.Builder()
          .dropOut(0.9)
          .nIn(featSize)
          .nOut(featSize / 2)
          .build(),
        "avg"
      )
      .addLayer(
        "output",
        new OutputLayer.Builder()
          .nIn(featSize / 2)
          .nOut(2)
          .lossFunction(LossFunction.MCXENT)
          .activation(Activation.SOFTMAX)
          .hasBias(false)
          .build,
        "dense"
      )
      .setOutputs("output")
      .build()

    val model = new ComputationGraph(modelConf)
    model.init()

    val trainingMaster =
      new ParameterAveragingTrainingMaster.Builder(batchSize)
        .rddTrainingApproach(RDDTrainingApproach.Direct)
        .build
    val sparkModel =
      new SparkComputationGraph(spark.sparkContext, model, trainingMaster)
    (0 until numEpochs).foreach(_ => sparkModel.fitMultiDataSet(trainingRdd))

  }

}
