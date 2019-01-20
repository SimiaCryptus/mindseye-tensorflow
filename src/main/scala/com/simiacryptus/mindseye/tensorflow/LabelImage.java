/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.simiacryptus.mindseye.tensorflow;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.simiacryptus.tensorflow.GraphModel;
import org.apache.commons.io.FileUtils;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.*;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.util.*;

/**
 * Sample use of the TensorFlow Java API to label images using a pre-trained model.
 */
public class LabelImage {
  private static final Logger logger = LoggerFactory.getLogger(LabelImage.class);

  public static void main(String[] args) throws URISyntaxException, NoSuchAlgorithmException, IOException, KeyManagementException {
    for (String imageFile : args) {
      LabelingNetwork loadNetwork = new LabelingNetwork();
      float[] labelProbabilities = predictImgFile(imageFile, loadNetwork.getProtobufSrc());
      int bestLabelIdx = maxIndex(labelProbabilities);
      System.out.println(
          String.format("BEST MATCH: %s (%.2f%% likely)",
              loadNetwork.getLabels().get(bestLabelIdx),
              labelProbabilities[bestLabelIdx] * 100f));
    }
  }

  public static float[] predictImgFile(String imageFile, byte[] graphDef) throws IOException {
    return predictImgBytes(FileUtils.readFileToByteArray(new File(imageFile)), graphDef);
  }

  public static float[] predictImgBytes(byte[] imageBytes, byte[] graphDef) {
    try (Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageBytes)) {
      return executeInceptionGraph(graphDef, image);
    }
  }

  private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
    logger.info("Tensorflow v" + TensorFlow.version());
    try (Graph graph = new Graph()) {
      GraphBuilder graphBuilder = new GraphBuilder(graph);

      // Some constants specific to the pre-trained model at:
      // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
      //
      // - The model was trained with images scaled to 224x224 pixels.
      // - The colors, represented as R, G, B in 1-byte each were converted to
      //   float using (value - Mean)/Scale.
      final int H = 224;
      final int W = 224;
      final float mean = 117f;
      final float scale = 1f;

      // Since the graph is being constructed once per execution here, we can use a constant for the
      // input image. If the graph were to be re-used for multiple input images, a placeholder would
      // have been more appropriate.
      final Output<String> input = graphBuilder.constant("input", imageBytes);
      final Output<Float> output =
          graphBuilder.div(
              graphBuilder.sub(
                  graphBuilder.resizeBilinear(
                      graphBuilder.expandDims(
                          graphBuilder.cast(graphBuilder.decodeJpeg(input, 3), Float.class),
                          graphBuilder.constant("make_batch", 0)),
                      graphBuilder.constant("size", new int[]{H, W})),
                  graphBuilder.constant("mean", mean)),
              graphBuilder.constant("scale", scale));
//      Ops ops = Ops.create(graph);
//      ops.applyGradientDescent(output,null,null).asOutput();

      try (Session s = new Session(graph)) {
        return s.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
      }
    }
  }


  private static float[] executeInceptionGraph(byte[] graphData, Tensor<Float> image) {
    describeGraph(loadAndSerialize(graphData));
    try (Graph g = new Graph()) {
      g.importGraphDef(graphData);
      describeGraph(g);
      try (
          Session s = new Session(g);
          Tensor<Float> result = s.runner().feed("input", image).fetch("output").run().get(0).expect(Float.class)
      ) {
        final long[] shape = result.shape();
        if (result.numDimensions() != 2 || shape[0] != 1) {
          throw new RuntimeException(
              String.format(
                  "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                  Arrays.toString(shape)));
        }
        return result.copyTo(new float[1][(int) shape[1]])[0];
      }
    }
  }


  public static void describeGraph(byte[] bytes) {
    System.out.println("Decoding GraphDef from " + bytes.length + " bytes");

    try {
      System.out.println("Model: " + new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT).writeValueAsString(new GraphModel(bytes).getChild("output")));
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
  }

  private static byte[] loadAndSerialize(byte[] graphData) {
    final byte[] bytes;
    try (Graph graph = new Graph()) {
      graph.importGraphDef(graphData);
      try {
        bytes = graph.toGraphDef();
      } catch (Throwable e) {
        e.printStackTrace();
        throw new RuntimeException(e);
      }
    }
    return bytes;
  }

  private static void describeGraph(Graph graph) {
    try {
      ArrayList<String> names = getNames(graph);
      for (Iterator<Operation> iter = graph.operations(); iter.hasNext(); ) {
        Operation op = iter.next();
        System.out.println(String.format("Operation %s (type %s) with %s outputs", op.name(), op.type(), op.numOutputs()));
        for (int i = 0; i < op.numOutputs(); i++) {
          Output<Object> output = op.output(i);
          System.out.println(String.format("Output %s (type %s, shape %s)", i, output.dataType().name(), output.shape().toString()));
        }
        for (String input : names) {
          try {
            System.out.println(String.format("Input List Length (%s): %s", input, op.inputListLength(input)));
          } catch (Throwable e) {
//          System.out.println(String.format("Input List Length (%s): %s", input, e.getMessage()));
          }
        }
      }
    } catch (Throwable e) {
      e.printStackTrace();
    }
  }

  @NotNull
  private static ArrayList<String> getNames(Graph graph) {
    ArrayList<String> names = new ArrayList<>();
    for (Iterator<Operation> iter = graph.operations(); iter.hasNext(); ) {
      names.add(iter.next().name());
    }
    return names;
  }

  public static int maxIndex(float[] probabilities) {
    int best = 0;
    for (int i = 1; i < probabilities.length; ++i) {
      if (probabilities[i] > probabilities[best]) {
        best = i;
      }
    }
    return best;
  }

}


