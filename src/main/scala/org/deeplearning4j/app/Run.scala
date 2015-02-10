package org.deeplearning4j.app

import org.nd4j.linalg.factory.Nd4j

/**
 * Created by agibsonccc on 2/9/15.
 */

class Run extends App {
  override def main(args: Array[String]) {
    //create a 5 length row vector
    val arr = Nd4j.zeros(5)
    //add 1 in place
    arr.addi(1)
    println(arr)
    //column vector
    val arrT = arr.transpose()



  }
}

