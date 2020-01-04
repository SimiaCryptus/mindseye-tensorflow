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
import com.simiacryptus.mindseye.layers.java.LayerTestBase;
import com.simiacryptus.mindseye.layers.tensorflow.BiasLayer;
import com.simiacryptus.mindseye.layers.tensorflow.MatMulLayer;
import com.simiacryptus.mindseye.layers.tensorflow.SoftmaxLayer;
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
class AtomicTFMnist {

  private static final boolean summarize = false;

  public static Layer network() {
    return network(new NullNotebookOutput());
  }

  public static Layer network(NotebookOutput log) {
    return log.eval(() -> {
      @Nonnull final PipelineNetwork pipeline = new PipelineNetwork();
      if (summarize)
        pipeline.add(new SummaryLayer("input"));
      pipeline
          .add(new MatMulLayer(new int[]{28, 28, 1}, new int[]{10}).set(() -> 0.001 * (Math.random() - 0.45)));
      if (summarize)
        pipeline.add(new SummaryLayer("matmul"));
      pipeline.add(new BiasLayer(10));
      if (summarize)
        pipeline.add(new SummaryLayer("bias"));
      pipeline.add(new SoftmaxLayer());
      if (summarize)
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
  }

}
