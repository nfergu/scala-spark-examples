# Spark and Data Science on the JVM



#Nd4j


Nd4j is a scientific computing framework on the JVM meant to emulate coding practices of numpy and matlab. JVM based code often involves loops and lacks ndarrays.  Nd4j is an interface to a number of [BLAS](http://www.netlib.org/blas/) implementations such as [jcublas](http://jcuda.org/) and [jblas](http://mikiobraun.github.io/jblas/)



##Spark


Deep learning4j spark uses parallel iterative reduce parameter averaging with training on datasets a mini batch size at a time based on partitioning. We expose an ml lib like interface to the algorithms. The general idea is as follows:

1. Instanitate a configuration the same way you would in deeplearning4j-core. 
 ```scala

    val conf = new NeuralNetConfiguration.Builder().momentum(0.9)
      .activationFunction(Activations.tanh()).weightInit(WeightInit.VI)
      .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
      .iterations(100).visibleUnit(RBM.VisibleUnit.GAUSSIAN)
      .hiddenUnit(RBM.HiddenUnit.RECTIFIED).stepFunction(new GradientStepFunction())
      .nIn(4).nOut(3).layerFactory(layerFactory)
      .list(3).hiddenLayerSizes(3, 2).`override`(classifierOverride )
      .build()


```


Setup a spark conf and context as normal

 ```scala
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
 ```



Contributions:

Scala community! One of the things we need is a scala wrapper for nd4j. Operator overloading for INDArray is a huge priority for us. 

We also need a scala wrapper for deeplearning4j. There are definitely better ways we could be doing things. Please subscribe to our google group:
http://groups.google.com/forum/#!forum/deeplearning4j

Submit issues:
https://github.com/SkymindIO/deeplearning4j/issues

Vote for our hadoop summit talk as well!

https://hadoopsummit.uservoice.com/forums/283261-data-science-and-hadoop/filters/top








