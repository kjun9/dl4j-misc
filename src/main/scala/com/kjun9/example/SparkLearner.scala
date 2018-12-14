package com.kjun9.example

import org.apache.spark.sql.SparkSession
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
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
    val numBatches = 1000
    val numEpochs = 5
    val random = new Random(0)

    // dummy training data with 2 input feature matrices
    val trainingData: Seq[api.MultiDataSet] = (0 until numBatches).map { feat =>
      val features =
        Array(Nd4j.create(Array.fill(batchSize * featSize)(random.nextDouble), Array(batchSize, featSize)))
      val labels =
        Array(Nd4j.create(Array.fill(batchSize)(random.nextDouble), Array(batchSize, 1)))
      new MultiDataSet(features, labels)
    }
    val trainingRdd = spark.sparkContext.parallelize(trainingData)

    // simple model
    val modelConf = new NeuralNetConfiguration.Builder()
      .updater(new Adam(0.01))
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .biasInit(0)
      .graphBuilder()
      .addInputs("input")
      .addLayer(
        "dense",
        new DenseLayer.Builder()
          .nIn(10)
          .nOut(10)
          .activation(Activation.RELU)
          .hasBias(true)
          .dropOut(0.9)
          .build,
        "input"
      )
      .addLayer(
        "output",
        new OutputLayer.Builder()
          .nIn(10)
          .nOut(1)
          .lossFunction(LossFunction.XENT)
          .activation(Activation.SIGMOID)
          .hasBias(false)
          .build,
        "dense"
      )
      .setOutputs("output")
      .build()

    val model = new ComputationGraph(modelConf)
    model.init()

    // training without spark works
//    (0 until numEpochs).foreach(_ =>
//      (0 until numBatches).foreach(i => model.fit(trainingData(i)))
//    )

    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(batchSize).build
    val sparkModel = new SparkComputationGraph(spark.sparkContext, model, trainingMaster)
    (0 until numEpochs).foreach(_ => sparkModel.fitMultiDataSet(trainingRdd))

  }

}
