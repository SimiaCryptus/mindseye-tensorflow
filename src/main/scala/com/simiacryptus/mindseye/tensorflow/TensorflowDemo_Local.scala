package com.simiacryptus.mindseye.tensorflow

import java.io.ByteArrayInputStream

import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner
import javax.imageio.ImageIO
import org.apache.commons.io.IOUtils
import org.apache.http.client.methods.HttpGet
import org.apache.http.impl.client.HttpClientBuilder

object TensorflowDemo_Local extends LocalRunner[Object] with NotebookRunner[Object] {
  override def apply(log: NotebookOutput): Object = {
    import LabelImage._
    val loadNetwork = new LabelingNetwork
    val client = HttpClientBuilder.create().build()
    for(keyword <- List("dog","cat","ship","city")) {
      val url = "https://loremflickr.com/320/240/" + keyword
      val bytes = IOUtils.toByteArray(client.execute(new HttpGet(url)).getEntity.getContent)
      log.p(log.jpg(ImageIO.read(new ByteArrayInputStream(bytes)), "Random Image"))
      log.run(()=>{
        val predictions = predictImgBytes(bytes, loadNetwork.getProtobufSrc).toList
        for(index <- predictions.zipWithIndex.sortBy(_._1).reverse.take(5).map(_._2))
          System.out.println(f"${loadNetwork.getLabels.get(index)} (${predictions(index) * 100.0}%.2f%% likely)")
      })
    }
    null
  }
}
