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

package com.simiacryptus.mindseye.examples.mnist;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.layers.tensorflow.SummaryLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.Graph;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public @com.simiacryptus.ref.lang.RefAware
class CudnnJavaMnist {

  private static final boolean tensorboard = false;

  public static Layer network() {
    return network(new NullNotebookOutput());
  }

  public static Layer network(NotebookOutput log) {
    return log.eval(() -> {
      @Nonnull final PipelineNetwork pipeline = new PipelineNetwork();
      if (tensorboard)
        pipeline.add(new SummaryLayer("input"));

      int bands1 = 64;
      pipeline.add(new SimpleConvolutionLayer(5, 5, 1 * bands1).set(() -> 0.001 * (Math.random() - 0.45)));
      pipeline.add(new com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer(bands1));
      pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      pipeline.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      if (tensorboard)
        pipeline.add(new SummaryLayer("layerout1"));

      int bands2 = 32;
      pipeline.add(new SimpleConvolutionLayer(5, 5, bands1 * bands2).set(() -> 0.001 * (Math.random() - 0.45)));
      pipeline.add(new com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer(bands2));
      pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      pipeline.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      if (tensorboard)
        pipeline.add(new SummaryLayer("layerout2"));

      pipeline.add(new AssertDimensionsLayer(7, 7, bands2));
      pipeline.add(new com.simiacryptus.mindseye.layers.cudnn.conv.FullyConnectedLayer(new int[]{7, 7, bands2},
          new int[]{1024}).set(() -> 0.001 * (Math.random() - 0.45)).explode());
      pipeline.add(new BiasLayer(1024));
      pipeline.add(new ReLuActivationLayer());
      if (tensorboard)
        pipeline.add(new SummaryLayer("layerout3"));

      PipelineNetwork stochasticTerminal = new PipelineNetwork(1);
      stochasticTerminal.add(BinaryNoiseLayer.maskLayer(Math.pow(0.5, 1.5)));
      stochasticTerminal
          .add(new FullyConnectedLayer(new int[]{1024}, new int[]{10}).set(() -> 0.001 * (Math.random() - 0.45)));
      stochasticTerminal.add(new BiasLayer(10));
      stochasticTerminal.add(new SoftmaxLayer());
      pipeline.add(new StochasticSamplingSubnetLayer(stochasticTerminal, 5));

      if (tensorboard)
        pipeline.add(new SummaryLayer("softmax"));
      return pipeline;
    });
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class MnistDemo extends MnistDemoBase {
    @Override
    protected byte[] getGraphDef() {
      return new Graph().toGraphDef();
    }

    @Override
    protected Layer buildModel(@Nonnull NotebookOutput log) {
      timeout = 15 * 60;
      log.p("This is a very simple model that performs basic logistic regression. "
          + "It is expected to be trainable to about 91% accuracy on MNIST.");
      return network(log);
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware
  class LayerTest extends LayerTestBase {

    @Nullable
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
    }

    public static @SuppressWarnings("unused")
    LayerTest[] addRefs(LayerTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(LayerTest::addRef)
          .toArray((x) -> new LayerTest[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{28, 28}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return network();
    }

    @Override
    public void run(@NotNull @Nonnull NotebookOutput log) {
      super.run(log);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    LayerTest addRef() {
      return (LayerTest) super.addRef();
    }

    @Override
    protected Layer lossLayer() {
      return new EntropyLossLayer();
    }
  }

}
