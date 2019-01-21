/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.tensorflow

import java.io.ByteArrayInputStream

import com.fasterxml.jackson.databind.{ObjectMapper, SerializationFeature}
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner
import com.simiacryptus.tensorflow.{GraphModel, InceptionClassifier}
import javax.imageio.ImageIO
import org.apache.commons.io.IOUtils
import org.apache.http.client.methods.HttpGet
import org.apache.http.impl.client.HttpClientBuilder

object TensorflowDemo_Local extends LocalRunner[Object] with NotebookRunner[Object] {
  override def apply(log: NotebookOutput): Object = {
//    val loadNetwork = new InceptionClassifier
//    val client = HttpClientBuilder.create().build()
//    log.eval(()=>{
//      new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT).writeValueAsString(new GraphModel(loadNetwork.getGraphDef))
//    })
//    log.eval(()=>{
//      loadNetwork.describeGraph()
//    })
//    for(keyword <- List("dog","cat","ship","city")) {
//      val url = "https://loremflickr.com/320/240/" + keyword
//      val bytes = IOUtils.toByteArray(client.execute(new HttpGet(url)).getEntity.getContent)
//      log.p(log.jpg(ImageIO.read(new ByteArrayInputStream(bytes)), "Random Image"))
//      log.run(()=>{
//        val predictions = loadNetwork.predictImgBytes(bytes).toList
//        for(index <- predictions.zipWithIndex.sortBy(_._1).reverse.take(5).map(_._2))
//          System.out.println(f"${loadNetwork.getLabels.get(index)} (${predictions(index) * 100.0}%.2f%% likely)")
//      })
//    }
//    null
  }
}
