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
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.layers.tensorflow.SummaryLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.Graph;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public class CudnnJavaMnist {

  private static final boolean tensorboard = false;

  public static Layer network() {
    return network(new NullNotebookOutput());
  }

  public static Layer network(NotebookOutput log) {
    return log.eval(() -> {
      @Nonnull
      final PipelineNetwork pipeline = new PipelineNetwork();
      if (tensorboard)
        pipeline.add(new SummaryLayer("input"));

      int bands1 = 64;
      SimpleConvolutionLayer temp_12_0001 = new SimpleConvolutionLayer(5, 5, 1 * bands1);
      RefUtil.freeRef(pipeline.add(temp_12_0001.set(() -> 0.001 * (Math.random() - 0.45))));
      if (null != temp_12_0001)
        temp_12_0001.freeRef();
      RefUtil.freeRef(pipeline.add(new ImgBandBiasLayer(bands1)));
      PoolingLayer temp_12_0002 = new PoolingLayer();
      RefUtil.freeRef(pipeline.add(temp_12_0002.setMode(PoolingLayer.PoolingMode.Max)));
      if (null != temp_12_0002)
        temp_12_0002.freeRef();
      RefUtil.freeRef(pipeline.add(new ActivationLayer(ActivationLayer.Mode.RELU)));
      if (tensorboard)
        pipeline.add(new SummaryLayer("layerout1"));

      int bands2 = 32;
      SimpleConvolutionLayer temp_12_0003 = new SimpleConvolutionLayer(5, 5, bands1 * bands2);
      RefUtil.freeRef(pipeline.add(temp_12_0003.set(() -> 0.001 * (Math.random() - 0.45))));
      if (null != temp_12_0003)
        temp_12_0003.freeRef();
      RefUtil.freeRef(pipeline.add(new ImgBandBiasLayer(bands2)));
      PoolingLayer temp_12_0004 = new PoolingLayer();
      RefUtil.freeRef(pipeline.add(temp_12_0004.setMode(PoolingLayer.PoolingMode.Max)));
      if (null != temp_12_0004)
        temp_12_0004.freeRef();
      RefUtil.freeRef(pipeline.add(new ActivationLayer(ActivationLayer.Mode.RELU)));
      if (tensorboard)
        pipeline.add(new SummaryLayer("layerout2"));

      RefUtil.freeRef(pipeline.add(new AssertDimensionsLayer(7, 7, bands2)));
      com.simiacryptus.mindseye.layers.cudnn.conv.FullyConnectedLayer temp_12_0005 = new com.simiacryptus.mindseye.layers.cudnn.conv.FullyConnectedLayer(
          new int[] { 7, 7, bands2 }, new int[] { 1024 });
      com.simiacryptus.mindseye.layers.cudnn.conv.FullyConnectedLayer temp_12_0007 = temp_12_0005
          .set(() -> 0.001 * (Math.random() - 0.45));
      RefUtil.freeRef(pipeline.add(temp_12_0007.explode()));
      if (null != temp_12_0007)
        temp_12_0007.freeRef();
      if (null != temp_12_0005)
        temp_12_0005.freeRef();
      RefUtil.freeRef(pipeline.add(new BiasLayer(1024)));
      RefUtil.freeRef(pipeline.add(new ReLuActivationLayer()));
      if (tensorboard)
        pipeline.add(new SummaryLayer("layerout3"));

      PipelineNetwork stochasticTerminal = new PipelineNetwork(1);
      RefUtil.freeRef(stochasticTerminal.add(BinaryNoiseLayer.maskLayer(Math.pow(0.5, 1.5))));
      FullyConnectedLayer temp_12_0006 = new FullyConnectedLayer(new int[] { 1024 }, new int[] { 10 });
      RefUtil.freeRef(stochasticTerminal.add(temp_12_0006.set(() -> 0.001 * (Math.random() - 0.45))));
      if (null != temp_12_0006)
        temp_12_0006.freeRef();
      RefUtil.freeRef(stochasticTerminal.add(new BiasLayer(10)));
      RefUtil.freeRef(stochasticTerminal.add(new SoftmaxLayer()));
      RefUtil.freeRef(pipeline
          .add(new StochasticSamplingSubnetLayer(stochasticTerminal == null ? null : stochasticTerminal.addRef(), 5)));

      if (null != stochasticTerminal)
        stochasticTerminal.freeRef();
      if (tensorboard)
        pipeline.add(new SummaryLayer("softmax"));
      return pipeline;
    });
  }

  public static class MnistDemo extends MnistDemoBase {
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

  public static class LayerTest extends LayerTestBase {

    @Nullable
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
    }

    public static @SuppressWarnings("unused") LayerTest[] addRefs(LayerTest[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(LayerTest::addRef).toArray((x) -> new LayerTest[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 28, 28 } };
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

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") LayerTest addRef() {
      return (LayerTest) super.addRef();
    }

    @Override
    protected Layer lossLayer() {
      return new EntropyLossLayer();
    }
  }

}
