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

public class NoiseJavaMnist {

  private static final boolean tensorboard = false;

  public static Layer network() {
    return network(new NullNotebookOutput());
  }

  public static Layer network(NotebookOutput log) {
    return log.eval(() -> {
      @Nonnull final PipelineNetwork pipeline = new PipelineNetwork();
      if (tensorboard)
        pipeline.add(new SummaryLayer("input"));

      int size1 = 100;
      pipeline.add(new FullyConnectedLayer(new int[]{28, 28, 1}, new int[]{size1})
            .set(() -> 0.001 * (Math.random() - 0.45)));
      if (tensorboard)
        pipeline.add(new SummaryLayer("multiply1"));
      pipeline.add(new BiasLayer(size1));
      if (tensorboard)
        pipeline.add(new SummaryLayer("bias1"));
      pipeline.add(new ReLuActivationLayer());
      pipeline.add(BinaryNoiseLayer.maskLayer(0.6));
      if (tensorboard)
        pipeline.add(new SummaryLayer("layerout1"));

      int size2 = 100;
      pipeline.add(new FullyConnectedLayer(new int[]{size1}, new int[]{size2}).set(() -> 0.001 * (Math.random() - 0.45)));
      if (tensorboard)
        pipeline.add(new SummaryLayer("multiply2"));
      pipeline.add(new BiasLayer(size2));
      if (tensorboard)
        pipeline.add(new SummaryLayer("bias2"));
      pipeline.add(new ReLuActivationLayer());
      pipeline.add(BinaryNoiseLayer.maskLayer(0.6));
      if (tensorboard)
        pipeline.add(new SummaryLayer("layerout2"));

      pipeline.add(new FullyConnectedLayer(new int[]{size2}, new int[]{10}).set(() -> 0.001 * (Math.random() - 0.45)));
      if (tensorboard)
        pipeline.add(new SummaryLayer("multiply3"));
      pipeline.add(new BiasLayer(10));
      if (tensorboard)
        pipeline.add(new SummaryLayer("bias3"));
      pipeline.add(new SoftmaxLayer());

      if (tensorboard)
        pipeline.add(new SummaryLayer("softmax"));
      return new StochasticSamplingSubnetLayer(pipeline, 5);
    });
  }

  public static class MnistDemo extends MnistDemoBase {
    @Override
    protected byte[] getGraphDef() {
      return new Graph().toGraphDef();
    }

    @Override
    protected Layer buildModel(@Nonnull NotebookOutput log) {
      timeout = 60 * 60;
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
  }

}
