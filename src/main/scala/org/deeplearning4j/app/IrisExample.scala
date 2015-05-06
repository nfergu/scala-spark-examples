package org.deeplearning4j.app

import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.models.featuredetectors.rbm.RBM
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.{Builder, ConfOverride}
import org.deeplearning4j.nn.layers.OutputLayer
import org.deeplearning4j.nn.layers.factory.LayerFactories
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.stepfunctions.GradientStepFunction
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.util.MLLibUtil
import org.nd4j.linalg.api.activation.Activations
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
 * Created by agibsonccc on 2/8/15.
 */
class IrisExample {

  def main(args: Array[String]) {
    val layerFactory = LayerFactories.getFactory(classOf[RBM])
    val classifierOverride = new ConfOverride {
      override def `override`(i: Int, builder: Builder):  Unit = {
        builder.activationFunction(Activations.softMaxRows)
        builder.layerFactory(LayerFactories.getFactory(classOf[OutputLayer]))
        builder.lossFunction(LossFunctions.LossFunction.MCXENT)
      }
    }


    val conf = new NeuralNetConfiguration.Builder().momentum(0.9)
      .activationFunction(Activations.tanh()).weightInit(WeightInit.VI)
      .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
      .iterations(100).visibleUnit(RBM.VisibleUnit.GAUSSIAN)
      .hiddenUnit(RBM.HiddenUnit.RECTIFIED).stepFunction(new GradientStepFunction())
      .nIn(4).nOut(3).layerFactory(layerFactory)
      .list(3).hiddenLayerSizes(3, 2).`override`(classifierOverride )
      .build()


    val network = new MultiLayerNetwork(conf)
    network.init()

    val sparkConf = new SparkConf().setMaster("local[8]")
      .set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, "false")
      .set("spark.akka.frameSize", "100").setAppName("mnist")

    val sc = new JavaSparkContext(new SparkContext(sparkConf))

    val d: DataSet = new IrisDataSetIterator(150, 150).next
    d.normalizeZeroMeanZeroUnitVariance
    d.shuffle
    val next: java.util.List[DataSet] = d.asList

    val data: JavaRDD[DataSet] = sc.parallelize(next)
    val examples = MLLibUtil.fromDataSet(sc,data).rdd
    val network2: MultiLayerNetwork = SparkDl4jMultiLayer.train(examples,conf)

    val predictModel = new SparkDl4jMultiLayer(sc,network2)


    val scoreAndLabels = examples.map { point =>
      val score = predictModel.predict(point.features).toArray(0)
      (score,point.label)
    }


    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()


  }

}
