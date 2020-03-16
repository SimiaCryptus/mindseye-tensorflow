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
import com.simiacryptus.ref.lang.RefIgnore;
import org.junit.After;
import org.junit.Ignore;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.TestInfo;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public class LRNLayerTest extends RawTFLayerTestBase {

  @RefIgnore
  private final TFLayerBase tfLayer = createTFLayer();

  public LRNLayerTest() {
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

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{2, 3, 20}};
  }

  @After
  public void cleanup() {
    super.cleanup();
    if (null != tfLayer)
      tfLayer.freeRef();
  }

  @Nonnull
  protected TFLayerBase createTFLayer() {
    LRNLayer lrnLayer = new LRNLayer();
    lrnLayer.setRadius(5);
    lrnLayer.setAlpha(1e-4f);
    lrnLayer.setBias((float) 2);
    return lrnLayer;
  }

}
