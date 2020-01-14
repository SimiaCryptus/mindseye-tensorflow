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

import com.google.gson.JsonObject;
import com.google.protobuf.InvalidProtocolBufferException;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.tensorflow.TFIO;
import com.simiacryptus.mindseye.lang.tensorflow.TFUtil;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.tensorflow.TensorboardEventWriter;
import com.simiacryptus.tensorflow.TensorflowUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.*;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.Summary;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.BiConsumer;

public abstract class TFLayerBase extends LayerBase {
  private static final Logger log = LoggerFactory.getLogger(TFLayer.class);
  @Nullable
  public static TensorboardEventWriter eventWriter = null;

  private final RefMap<String, Tensor> weights = new RefHashMap<>();

  public TFLayerBase(@Nonnull JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    RefSet<String> dataKeys = getDataKeys(json);
    for (String key : dataKeys) {
      RefMap<String, com.simiacryptus.mindseye.lang.Tensor> temp_00_0010 = this.getWeights();
      assert temp_00_0010 != null;
      RefUtil.freeRef(temp_00_0010.put(key, Tensor.fromJson(json.get(key), rs)));
      temp_00_0010.freeRef();
    }
    dataKeys.freeRef();
  }

  public TFLayerBase(@Nullable RefMap<String, Tensor> states) {
    RefMap<String, com.simiacryptus.mindseye.lang.Tensor> temp_00_0011 = this.getWeights();
    assert temp_00_0011 != null;
    temp_00_0011.putAll(states == null ? null : states.addRef());
    temp_00_0011.freeRef();
    if (null != states)
      states.freeRef();
  }

  public abstract GraphDef getGraphDef();

  @Nullable
  public abstract List<String> getInputNodes();

  public abstract String getOutputNode();

  @Nullable
  public abstract String getSummaryOut();

  @Nullable
  public RefMap<String, Tensor> getWeights() {
    return weights == null ? null : weights.addRef();
  }

  @Nullable
  public static @SuppressWarnings("unused")
  TFLayerBase[] addRefs(@Nullable TFLayerBase[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TFLayerBase::addRef).toArray((x) -> new TFLayerBase[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  TFLayerBase[][] addRefs(@Nullable TFLayerBase[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TFLayerBase::addRefs).toArray((x) -> new TFLayerBase[x][]);
  }

  @Nonnull
  public TFLayer asConstLayer() {
    return new TFLayer(constGraph().toByteArray(), new RefHashMap<>(), getOutputNode(),
        getInputNodes().toArray(new String[]{}));
  }

  public @Nonnull
  GraphDef constGraph() {
    return TFUtil.implantConstants(getGraphDef(), getWeights());
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    JsonObject json = getJsonStub();
    RefMap<String, com.simiacryptus.mindseye.lang.Tensor> temp_00_0014 = getWeights();
    assert temp_00_0014 != null;
    RefSet<Map.Entry<String, Tensor>> entries = temp_00_0014.entrySet();
    entries.forEach(entry -> {
      Tensor temp_00_0015 = entry.getValue();
      json.add(entry.getKey(), temp_00_0015.getJson(resources, dataSerializer));
      temp_00_0015.freeRef();
    });
    entries.freeRef();
    temp_00_0014.freeRef();
    return json;
  }

  @Nullable
  @Override
  public RefList<double[]> state() {
    RefMap<String, com.simiacryptus.mindseye.lang.Tensor> temp_00_0017 = getWeights();
    assert temp_00_0017 != null;
    RefCollection<Tensor> values = temp_00_0017.values();
    RefList<double[]> temp_00_0016 = values.stream().map(x -> {
      double[] temp_00_0002 = x.getData();
      x.freeRef();
      return temp_00_0002;
    }).collect(RefCollectors.toList());
    values.freeRef();
    temp_00_0017.freeRef();
    return temp_00_0016;
  }

  @Nullable
  @Override
  public Result eval(@Nullable Result... inputs) {
    TFSession tfsession = new TFSession(TFLayerBase.this.addRef());
    Result temp_00_0003 = eval(tfsession.addRef(), Result.addRefs(inputs));
    if (null != inputs)
      ReferenceCounting.freeRefs(inputs);
    tfsession.freeRef();
    return temp_00_0003;
  }

  public void close() {
  }

  public boolean invertWeights() {
    return true;
  }

  public @Nonnull
  GraphDef getConstGraph(GraphDef graphDef) {
    return TFUtil.implantConstants(graphDef, getWeights());
  }

  public void _free() {
    if (null != weights)
      weights.freeRef();
    close();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  TFLayerBase addRef() {
    return (TFLayerBase) super.addRef();
  }

  @Nonnull
  Result eval(@Nonnull TFSession tfsession, @Nonnull Result... inputs) {
    RefMap<String, com.simiacryptus.mindseye.lang.Tensor> temp_00_0018 = getWeights();
    assert temp_00_0018 != null;
    RefSet<String> temp_00_0019 = temp_00_0018.keySet();
    RefList<String> stateNames = temp_00_0019.stream().collect(RefCollectors.toList());
    temp_00_0019.freeRef();
    temp_00_0018.freeRef();
    Session.Runner runner = tfsession.session.runner();
    RefArrayList<org.tensorflow.Tensor<?>> tensors = new RefArrayList<>();
    RefMap<String, com.simiacryptus.mindseye.lang.Tensor> temp_00_0020 = getWeights();
    temp_00_0020.forEach(RefUtil
        .wrapInterface((BiConsumer<? super String, ? super com.simiacryptus.mindseye.lang.Tensor>) (nodeName, data) -> {
          @Nonnull org.tensorflow.Tensor<? extends Number> tensor;
          if (floatInputs(nodeName)) {
            tensor = TFIO.getFloatTensor(data == null ? null : data.addRef(), invertWeights());
          } else {
            tensor = TFIO.getDoubleTensor(data == null ? null : data.addRef(), invertWeights());
          }
          if (null != data)
            data.freeRef();
          runner.feed(nodeName, tensor);
          tensors.add(tensor);
        }, tensors.addRef()));
    temp_00_0020.freeRef();
    final List<String> inputNodes = getInputNodes();
    assert inputNodes != null;
    for (int i = 0; i < inputNodes.size(); i++) {
      String inputNode = inputNodes.get(i);
      TensorList data = inputs[i].getData();
      @Nonnull org.tensorflow.Tensor<? extends Number> tensor;
      if (floatInputs(inputNode)) {
        tensor = TFIO.getFloatTensor(data.addRef());
      } else {
        tensor = TFIO.getDoubleTensor(data.addRef());
      }
      data.freeRef();
      runner.feed(inputNode, tensor);
      tensors.add(tensor);
    }
    runner.fetch(getOutputNode());
    boolean summaryOut = null != eventWriter && null != getSummaryOut() && !getSummaryOut().isEmpty();
    int fwdFetches;
    if (summaryOut) {
      fwdFetches = 2;
      runner.fetch(getSummaryOut());
    } else {
      fwdFetches = 1;
    }
    Session.Run fwd;
    try {
      fwd = runner.runAndFetchMetadata();
    } catch (IllegalArgumentException e) {
      throw e;
    }
    TensorArray resultData;
    {
      org.tensorflow.Tensor<?> tensor = fwd.outputs.get(0);
      resultData = TFIO.getTensorList(tensor);
      tensors.add(tensor);
    }
    if (summaryOut) {
      final Summary summary;
      try {
        summary = Summary.parseFrom(fwd.outputs.get(1).expect(String.class).bytesValue());
      } catch (InvalidProtocolBufferException e) {
        throw new RuntimeException(e);
      }
      try {
        if (null != eventWriter)
          eventWriter.write(summary);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
    try {
      return new Result(resultData, (new Result.Accumulator() {
        {
          tfsession.addRef();
          Result.addRefs(inputs);
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> deltaBuffer, @Nullable TensorList deltaSignal) {
          RefArrayList<org.tensorflow.Tensor<?>> feedbacktensors = new RefArrayList<>();
          Output<?>[] gradients = tfsession.getGradients();
          String deltaOperation = TFLayerBase.this.getOutputNode() + "_delta";
          if (TFLayerBase.this.floatInputs(deltaOperation)) {
            org.tensorflow.Tensor<Float> tensor = TFIO
                .getFloatTensor(deltaSignal == null ? null : deltaSignal.addRef());
            runner.feed(deltaOperation, tensor);
            feedbacktensors.add(tensor);
          } else {
            org.tensorflow.Tensor<Double> tensor = TFIO
                .getDoubleTensor(deltaSignal == null ? null : deltaSignal.addRef());
            runner.feed(deltaOperation, tensor);
            feedbacktensors.add(tensor);
          }
          if (null != deltaSignal)
            deltaSignal.freeRef();
          RefArrays.stream(gradients).forEach(runner::fetch);
          Session.Run back = runner.runAndFetchMetadata();
          for (int i = 0; i < inputs.length; i++) {
            org.tensorflow.Tensor<?> tensor = back.outputs.get(fwdFetches + i);
            Result.Accumulator temp_00_0023 = inputs[i].getAccumulator();
            assert temp_00_0023 != null;
            temp_00_0023.accept(deltaBuffer == null ? null : deltaBuffer.addRef(), TFIO.getTensorList(tensor));
            temp_00_0023.freeRef();
            feedbacktensors.add(tensor);
          }
          for (int i = 0; i < stateNames.size(); i++) {
            String weightNodeName = stateNames.get(i);
            RefMap<String, com.simiacryptus.mindseye.lang.Tensor> temp_00_0024 = TFLayerBase.this.getWeights();
            assert deltaBuffer != null;
            Delta<UUID> uuidDelta = deltaBuffer.get(
                UUID.nameUUIDFromBytes((TFLayerBase.this.getId() + "_" + weightNodeName).getBytes()),
                temp_00_0024.get(weightNodeName));
            temp_00_0024.freeRef();
            org.tensorflow.Tensor<Number> numberTensor = (org.tensorflow.Tensor<Number>) back.outputs
                .get(i + fwdFetches + TFLayerBase.this.getInputNodes().size());
            Tensor t;
            if (numberTensor.dataType() == DataType.FLOAT) {
              t = TFIO.getTensor(numberTensor.expect(Float.class), TFLayerBase.this.invertWeights());
            } else {
              t = TFIO.getTensor(numberTensor.expect(Double.class), TFLayerBase.this.invertWeights());
            }
            assert uuidDelta != null;
            RefUtil.freeRef(uuidDelta.addInPlace(t.getData()));
            t.freeRef();
            uuidDelta.freeRef();
          }
          if (null != deltaBuffer)
            deltaBuffer.freeRef();
          feedbacktensors.stream().forEach(org.tensorflow.Tensor::close);
          feedbacktensors.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          ReferenceCounting.freeRefs(inputs);
          tfsession.freeRef();
          stateNames.freeRef();
        }
      }));
    } finally {
      ReferenceCounting.freeRefs(inputs);
      tfsession.freeRef();
      resultData.freeRef();
      tensors.freeRef();
      if (null != stateNames)
        stateNames.freeRef();
    }
  }

  @Nonnull
  protected abstract RefSet<String> getDataKeys(JsonObject json);

  protected boolean floatInputs(String key) {
    return false;
  }

  static class TFSession extends ReferenceCountingBase {
    @Nonnull
    public final Graph graph;
    public final Singleton<Output<?>[]> outputSingleton = new Singleton<>();
    @Nonnull
    public final Session session;
    @Nullable
    private final TFLayerBase parent;

    public TFSession(@Nullable TFLayerBase parent) {
      this.graph = new Graph();
      TFLayerBase temp_00_0001 = parent == null ? null : parent.addRef();
      this.parent = temp_00_0001 == null ? null : temp_00_0001.addRef();
      if (null != temp_00_0001)
        temp_00_0001.freeRef();
      assert parent != null;
      GraphDef graphDef = parent.getGraphDef();
      parent.freeRef();
      TensorflowUtil.validate(graphDef);
      graph.importGraphDef(graphDef.toByteArray());
      this.session = new Session(graph);
    }

    @Nonnull
    public Output<?>[] getGradients() {
      return outputSingleton.getOrInit(() -> {
        assert parent != null;
        RefMap<String, com.simiacryptus.mindseye.lang.Tensor> temp_00_0026 = parent.getWeights();
        assert temp_00_0026 != null;
        RefSet<String> temp_00_0027 = temp_00_0026.keySet();
        RefList<String> stateNames = temp_00_0027.stream().collect(RefCollectors.toList());
        temp_00_0027.freeRef();
        temp_00_0026.freeRef();
        Ops ops = Ops.create(graph);
        String deltaOpName = parent.getOutputNode() + "_delta";
        Class<? extends Number> dtype = parent.floatInputs(deltaOpName) ? Float.class : Double.class;
        ops.withName(deltaOpName).placeholder(dtype, Placeholder.shape(Shape.unknown()));
        Output<?>[] temp_00_0007 = graph.addGradients("gradient",
            new Output[]{TensorflowUtil.find(graph, parent.getOutputNode()).output(0)},
            RefStream.concat(parent.getInputNodes().stream(), stateNames.stream())
                .map(n -> TensorflowUtil.find(graph, n).output(0)).toArray(i -> new Output[i]),
            new Output[]{TensorflowUtil.find(graph, deltaOpName).output(0)});
        stateNames.freeRef();
        return temp_00_0007;
      });
    }

    @Nullable
    public static @SuppressWarnings("unused")
    TFSession[] addRefs(@Nullable TFSession[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(TFSession::addRef).toArray((x) -> new TFSession[x]);
    }

    public void _free() {
      if (null != parent)
        parent.freeRef();
      new Thread(() -> {
        session.close();
        graph.close();
      }).start();
      super._free();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    TFSession addRef() {
      return (TFSession) super.addRef();
    }
  }

}
