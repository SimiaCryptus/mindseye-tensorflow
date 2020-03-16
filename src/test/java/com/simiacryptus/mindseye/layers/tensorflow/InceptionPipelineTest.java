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

import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.mindseye.util.TFConverter;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.tensorflow.ImageNetworkPipeline;
import org.junit.Ignore;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.TestInfo;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public abstract class InceptionPipelineTest extends TFLayerTestBase {
  public static final RefList<TFLayer> layers = TFConverter.getLayers(ImageNetworkPipeline.inception5h());

  public InceptionPipelineTest() {
    this.testingBatchSize = 5;
  }

  @Nullable
  @Override
  public BatchingTester getBatchingTester() {
    return getBatchingTester(1e-2, false, this.testingBatchSize);
  }

  @Override
  @Disabled
  public void derivativeTest(TestInfo testInfo) {
    super.derivativeTest(testInfo);
  }

  @Override
  @Disabled
  public void trainingTest(TestInfo testInfo) {
    super.trainingTest(testInfo);
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

  }

}
