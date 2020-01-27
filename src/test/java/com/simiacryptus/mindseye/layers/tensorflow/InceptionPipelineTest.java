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

package com.simiacryptus.mindseye.layers.tensorflow;

import com.simiacryptus.mindseye.util.TFConverter;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.tensorflow.ImageNetworkPipeline;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract class InceptionPipelineTest extends TFLayerTestBase {
  public static final RefList<TFLayer> layers = TFConverter.getLayers(ImageNetworkPipeline.inception5h());

  public InceptionPipelineTest() {
    validateDifferentials = false;
    testTraining = false;
    this.testingBatchSize = 5;
  }

  public @SuppressWarnings("unused")
  void _free() { super._free(); }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  InceptionPipelineTest addRef() {
    return (InceptionPipelineTest) super.addRef();
  }

  public static class Layer0 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{320, 240, 3}};
    }

    public @Nonnull
    TFLayerBase createTFLayer() {
      return layers.get(0);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Layer0 addRef() {
      return (Layer0) super.addRef();
    }
  }

  public static class Layer1 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{160, 120, 64}};
    }

    public @Nonnull
    TFLayerBase createTFLayer() {
      return layers.get(1);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Layer1 addRef() {
      return (Layer1) super.addRef();
    }
  }

  public static class Layer2 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{80, 60, 192}};
    }

    public @Nonnull
    TFLayerBase createTFLayer() {
      return layers.get(2);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Layer2 addRef() {
      return (Layer2) super.addRef();
    }
  }

  public static class Layer3 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{40, 30, 256}};
    }

    public @Nonnull
    TFLayerBase createTFLayer() {
      return layers.get(3);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Layer3 addRef() {
      return (Layer3) super.addRef();
    }
  }

  public static class Layer4 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{40, 30, 480}};
    }

    public @Nonnull
    TFLayerBase createTFLayer() {
      return layers.get(4);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Layer4 addRef() {
      return (Layer4) super.addRef();
    }
  }

  public static class Layer5 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{20, 15, 508}};
    }

    public @Nonnull
    TFLayerBase createTFLayer() {
      return layers.get(5);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Layer5 addRef() {
      return (Layer5) super.addRef();
    }
  }

  public static class Layer6 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{20, 15, 512}};
    }

    public @Nonnull
    TFLayerBase createTFLayer() {
      return layers.get(6);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Layer6 addRef() {
      return (Layer6) super.addRef();
    }
  }

  public static class Layer7 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{20, 15, 512}};
    }

    public @Nonnull
    TFLayerBase createTFLayer() {
      return layers.get(7);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Layer7 addRef() {
      return (Layer7) super.addRef();
    }
  }

  public static class Layer8 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{20, 15, 528}};
    }

    public @Nonnull
    TFLayerBase createTFLayer() {
      return layers.get(8);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Layer8 addRef() {
      return (Layer8) super.addRef();
    }
  }

  public static class Layer9 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{20, 15, 832}};
    }

    public @Nonnull
    TFLayerBase createTFLayer() {
      return layers.get(9);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Layer9 addRef() {
      return (Layer9) super.addRef();
    }
  }

  public static class Layer10 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{10, 8, 832}};
    }

    public @Nonnull
    TFLayerBase createTFLayer() {
      return layers.get(10);
    }

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Layer10 addRef() {
      return (Layer10) super.addRef();
    }
  }

}
