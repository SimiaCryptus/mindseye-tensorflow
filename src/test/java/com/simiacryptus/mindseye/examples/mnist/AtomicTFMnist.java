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
import com.simiacryptus.mindseye.layers.tensorflow.BiasLayer;
import com.simiacryptus.mindseye.layers.tensorflow.MatMulLayer;
import com.simiacryptus.mindseye.layers.tensorflow.SoftmaxLayer;
import com.simiacryptus.mindseye.layers.tensorflow.SummaryLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.LayerTestBase;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import org.tensorflow.Graph;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public class AtomicTFMnist {

  private static final boolean summarize = false;

  public static Layer network() {
    return network(new NullNotebookOutput());
  }

  public static Layer network(@Nonnull NotebookOutput log) {
    return log.eval(() -> {
      @Nonnull final PipelineNetwork pipeline = new PipelineNetwork();
      if (summarize)
        pipeline.add(new SummaryLayer("input")).freeRef();
      MatMulLayer temp_14_0001 = new MatMulLayer(new int[]{28, 28, 1}, new int[]{10});
      temp_14_0001.set(() -> 0.001 * (Math.random() - 0.45));
      RefUtil.freeRef(pipeline.add(temp_14_0001.addRef()));
      temp_14_0001.freeRef();
      if (summarize)
        pipeline.add(new SummaryLayer("matmul")).freeRef();
      RefUtil.freeRef(pipeline.add(new BiasLayer(10)));
      if (summarize)
        pipeline.add(new SummaryLayer("bias")).freeRef();
      RefUtil.freeRef(pipeline.add(new SoftmaxLayer()));
      if (summarize)
        pipeline.add(new SummaryLayer("softmax")).freeRef();
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
