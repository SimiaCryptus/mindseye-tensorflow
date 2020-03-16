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
import com.simiacryptus.mindseye.test.LayerTestBase;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import org.tensorflow.Graph;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public class NoiseJavaMnist {

  private static final boolean tensorboard = false;

  public static Layer network() {
    return network(new NullNotebookOutput());
  }

  public static Layer network(@Nonnull NotebookOutput log) {
    return log.eval(() -> {
      @Nonnull final PipelineNetwork pipeline = new PipelineNetwork();
      if (tensorboard)
        pipeline.add(new SummaryLayer("input")).freeRef();

      int size1 = 100;
      FullyConnectedLayer temp_04_0002 = new FullyConnectedLayer(new int[]{28, 28, 1}, new int[]{size1});
      temp_04_0002.set(() -> 0.001 * (Math.random() - 0.45));
      RefUtil.freeRef(pipeline.add(temp_04_0002.addRef()));
      temp_04_0002.freeRef();
      if (tensorboard)
        pipeline.add(new SummaryLayer("multiply1")).freeRef();
      RefUtil.freeRef(pipeline.add(new BiasLayer(size1)));
      if (tensorboard)
        pipeline.add(new SummaryLayer("bias1")).freeRef();
      RefUtil.freeRef(pipeline.add(new ReLuActivationLayer()));
      RefUtil.freeRef(pipeline.add(BinaryNoiseLayer.maskLayer(0.6)));
      if (tensorboard)
        pipeline.add(new SummaryLayer("layerout1")).freeRef();

      int size2 = 100;
      FullyConnectedLayer temp_04_0003 = new FullyConnectedLayer(new int[]{size1}, new int[]{size2});
      temp_04_0003.set(() -> 0.001 * (Math.random() - 0.45));
      RefUtil.freeRef(pipeline.add(temp_04_0003.addRef()));
      temp_04_0003.freeRef();
      if (tensorboard)
        pipeline.add(new SummaryLayer("multiply2")).freeRef();
      RefUtil.freeRef(pipeline.add(new BiasLayer(size2)));
      if (tensorboard)
        pipeline.add(new SummaryLayer("bias2")).freeRef();
      RefUtil.freeRef(pipeline.add(new ReLuActivationLayer()));
      RefUtil.freeRef(pipeline.add(BinaryNoiseLayer.maskLayer(0.6)));
      if (tensorboard)
        pipeline.add(new SummaryLayer("layerout2")).freeRef();

      FullyConnectedLayer temp_04_0004 = new FullyConnectedLayer(new int[]{size2}, new int[]{10});
      temp_04_0004.set(() -> 0.001 * (Math.random() - 0.45));
      RefUtil.freeRef(pipeline.add(temp_04_0004.addRef()));
      temp_04_0004.freeRef();
      if (tensorboard)
        pipeline.add(new SummaryLayer("multiply3")).freeRef();
      RefUtil.freeRef(pipeline.add(new BiasLayer(10)));
      if (tensorboard)
        pipeline.add(new SummaryLayer("bias3")).freeRef();
      RefUtil.freeRef(pipeline.add(new SoftmaxLayer()));

      if (tensorboard)
        pipeline.add(new SummaryLayer("softmax")).freeRef();
      return new StochasticSamplingSubnetLayer(pipeline,
          5);
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

    @Nonnull
    @Override
    public Layer getLayer() {
      return network();
    }

    @Nullable
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{28, 28}};
    }

  }

}
