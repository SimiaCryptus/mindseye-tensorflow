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

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.protobuf.InvalidProtocolBufferException;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.tensorflow.TFIO;
import com.simiacryptus.mindseye.lang.tensorflow.TFUtil;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.tensorflow.TensorboardEventWriter;
import com.simiacryptus.tensorflow.TensorflowUtil;
import com.simiacryptus.util.Util;
import org.jetbrains.annotations.NotNull;
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
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

public abstract class TFLayerBase extends LayerBase {
  private static final Logger log = LoggerFactory.getLogger(TFLayer.class);
  @Nullable
  public static TensorboardEventWriter eventWriter = null;

  private final RefMap<String, Tensor> weights = new RefHashMap<>();

  public TFLayerBase(@Nonnull JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    Set<String> dataKeys = getDataKeys(json);
    for (String key : dataKeys) {
      RefMap<String, com.simiacryptus.mindseye.lang.Tensor> weights = this.getWeights();
      assert weights != null;
      RefUtil.freeRef(weights.put(key, Tensor.fromJson(json.get(key), rs)));
      weights.freeRef();
    }
  }

  public TFLayerBase(@Nullable RefMap<String, Tensor> states) {
    RefMap<String, com.simiacryptus.mindseye.lang.Tensor> weights = this.getWeights();
    assert weights != null;
    weights.putAll(states);
    weights.freeRef();
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
    RefMap<String, com.simiacryptus.mindseye.lang.Tensor> weights = getWeights();
    assert weights != null;
    weights.forEach((key, tensor) -> {
      JsonElement tensorJson = tensor.getJson(resources, dataSerializer);
      tensor.freeRef();
      json.add(key, tensorJson);
    });
    weights.freeRef();
    return json;
  }

  @Nullable
  @Override
  public RefList<double[]> state() {
    RefCollection<Tensor> values = weights.values();
    RefList<double[]> dataList = values.stream().map(x -> {
      try {
        return x.getData();
      } finally {
        x.freeRef();
      }
    }).collect(RefCollectors.toList());
    values.freeRef();
    return dataList;
  }

  @Nullable
  @Override
  public Result eval(@Nullable Result... inputs) {
    return eval(new TFSession(addRef()), inputs);
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
    RefMap<String, com.simiacryptus.mindseye.lang.Tensor> weights = getWeights();
    assert weights != null;
    RefSet<String> keySet = weights.keySet();
    RefList<String> stateNames = keySet.stream().collect(RefCollectors.toList());
    keySet.freeRef();
    Session.Runner runner = tfsession.session.runner();
    RefArrayList<org.tensorflow.Tensor<?>> tensors = setTensors(runner, weights, RefUtil.addRef(inputs));
    boolean summaryOut = run(runner);
    TensorArray resultData = getOutput(runner, tensors, summaryOut);
    Accumulator accumulator = new Accumulator(runner, summaryOut ? 2 : 1, stateNames, this.getId(), this.getWeights(),
        this.getOutputNode(), this.invertWeights(), this.getInputNodes(),
        this.floatInputs(this.getOutputNode() + "_delta"), tfsession.getGradients(), tfsession, inputs);
    return new Result(resultData, accumulator);
  }

  @Nonnull
  protected abstract Set<String> getDataKeys(JsonObject json);

  protected boolean floatInputs(String key) {
    return false;
  }

  private boolean run(Session.Runner runner) {
    runner.fetch(getOutputNode());
    boolean summaryOut = null != eventWriter && null != getSummaryOut() && !getSummaryOut().isEmpty();
    if (summaryOut) {
      runner.fetch(getSummaryOut());
    }
    return summaryOut;
  }

  @NotNull
  private RefArrayList<org.tensorflow.Tensor<?>> setTensors(Session.Runner runner, RefMap<String, Tensor> weights,
                                                            @Nonnull Result[] inputs) {
    RefArrayList<org.tensorflow.Tensor<?>> tensors = new RefArrayList<>();
    weights.forEach((nodeName, data) -> {
      @Nonnull
      org.tensorflow.Tensor<? extends Number> tensor;
      boolean invertRanks = invertWeights();
      if (floatInputs(nodeName)) {
        tensor = TFIO.getFloatTensor(data, invertRanks);
      } else {
        tensor = TFIO.getDoubleTensor(data, invertRanks);
      }
      runner.feed(nodeName, tensor);
      tensors.add(tensor);
    });
    weights.freeRef();
    final List<String> inputNodes = getInputNodes();
    assert inputNodes != null;
    for (int i = 0; i < inputNodes.size(); i++) {
      String inputNode = inputNodes.get(i);
      TensorList data = inputs[i].getData();
      @Nonnull
      org.tensorflow.Tensor<? extends Number> tensor;
      if (floatInputs(inputNode)) {
        tensor = TFIO.getFloatTensor(data, true);
      } else {
        tensor = TFIO.getDoubleTensor(data, true);
      }
      runner.feed(inputNode, tensor);
      tensors.add(tensor);
    }
    RefUtil.freeRef(inputs);
    return tensors;
  }

  @NotNull
  private TensorArray getOutput(Session.Runner runner, RefArrayList<org.tensorflow.Tensor<?>> tensors,
                                boolean summaryOut) {
    Session.Run fwd;
    try {
      fwd = runner.runAndFetchMetadata();
    } catch (IllegalArgumentException e) {
      throw e;
    }
    org.tensorflow.Tensor<?> tensor = fwd.outputs.get(0);
    TensorArray resultData = TFIO.getTensorList(tensor);
    tensors.add(tensor);
    tensors.freeRef();
    if (summaryOut) {
      final Summary summary;
      try {
        summary = Summary.parseFrom(fwd.outputs.get(1).expect(String.class).bytesValue());
      } catch (InvalidProtocolBufferException e) {
        throw Util.throwException(e);
      }
      try {
        if (null != eventWriter)
          eventWriter.write(summary);
      } catch (IOException e) {
        throw Util.throwException(e);
      }
    }
    return resultData;
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
      this.parent = parent;
      GraphDef graphDef = this.parent.getGraphDef();
      TensorflowUtil.validate(graphDef);
      graph.importGraphDef(graphDef.toByteArray());
      this.session = new Session(graph);
    }

    @Nonnull
    public Output<?>[] getGradients() {
      return outputSingleton.getOrInit(() -> {
        assert parent != null;
        RefMap<String, com.simiacryptus.mindseye.lang.Tensor> weights = parent.getWeights();
        assert weights != null;
        RefSet<String> keySet = weights.keySet();
        RefList<String> stateNames = keySet.stream().collect(RefCollectors.toList());
        keySet.freeRef();
        weights.freeRef();
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

    public void _free() {
      if (null != parent)
        parent.freeRef();
      new Thread(() -> {
        session.close();
        graph.close();
      }).start();
      outputSingleton.freeRef();
      super._free();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    TFSession addRef() {
      return (TFSession) super.addRef();
    }
  }

  private static class Accumulator extends Result.Accumulator {

    private final Session.Runner runner;
    private final int fwdFetches;
    private final RefList<String> stateNames;
    private final Result[] inputs;
    private final TFSession tfsession;
    private RefMap<String, Tensor> weights;
    private String outputNode;
    private UUID id;
    private boolean invertRanks;
    private List<String> inputNodes;
    private boolean floatInputs;
    private Output<?>[] gradients;

    public Accumulator(Session.Runner runner, int fwdFetches, RefList<String> stateNames, UUID id,
                       RefMap<String, Tensor> weights, String outputNode, boolean invertRanks, List<String> inputNodes,
                       boolean floatInputs, Output<?>[] gradients, TFSession tfsession, Result... inputs) {
      this.runner = runner;
      this.fwdFetches = fwdFetches;
      this.stateNames = stateNames;
      this.inputs = inputs;
      this.weights = weights;
      this.outputNode = outputNode;
      this.id = id;
      this.invertRanks = invertRanks;
      this.inputNodes = inputNodes;
      this.floatInputs = floatInputs;
      this.gradients = gradients;
      this.tfsession = tfsession;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> deltaBuffer, @Nullable TensorList deltaSignal) {
      RefArrayList<org.tensorflow.Tensor<?>> feedbacktensors = new RefArrayList<>();
      Output<?>[] gradients = this.gradients;
      if (floatInputs) {
        org.tensorflow.Tensor<Float> tensor = TFIO.getFloatTensor(deltaSignal == null ? null : deltaSignal.addRef());
        runner.feed(outputNode + "_delta", tensor);
        feedbacktensors.add(tensor);
      } else {
        org.tensorflow.Tensor<Double> tensor = TFIO.getDoubleTensor(deltaSignal == null ? null : deltaSignal.addRef());
        runner.feed(outputNode + "_delta", tensor);
        feedbacktensors.add(tensor);
      }
      if (null != deltaSignal)
        deltaSignal.freeRef();
      RefArrays.stream(gradients).forEach(output -> runner.fetch(output));
      Session.Run back = runner.runAndFetchMetadata();
      for (int i = 0; i < inputs.length; i++) {
        org.tensorflow.Tensor<?> tensor = back.outputs.get(fwdFetches + i);
        Result.Accumulator accumulator = inputs[i].getAccumulator();
        assert accumulator != null;
        accumulator.accept(deltaBuffer == null ? null : deltaBuffer.addRef(), TFIO.getTensorList(tensor));
        accumulator.freeRef();
        feedbacktensors.add(tensor);
      }
      for (int i = 0; i < stateNames.size(); i++) {
        String weightNodeName = stateNames.get(i);
        assert deltaBuffer != null;
        Delta<UUID> uuidDelta = deltaBuffer.get(UUID.nameUUIDFromBytes((id + "_" + weightNodeName).getBytes()),
            weights.get(weightNodeName));
        org.tensorflow.Tensor<Number> numberTensor = (org.tensorflow.Tensor<Number>) back.outputs
            .get(i + fwdFetches + inputNodes.size());
        final Tensor t;
        if (numberTensor.dataType() == DataType.FLOAT) {
          t = TFIO.getTensor(numberTensor.expect(Float.class), invertRanks);
        } else {
          t = TFIO.getTensor(numberTensor.expect(Double.class), invertRanks);
        }
        assert uuidDelta != null;
        uuidDelta.addInPlace(t);
        uuidDelta.freeRef();
      }
      if (null != deltaBuffer)
        deltaBuffer.freeRef();
      feedbacktensors.stream().forEach(tensor -> tensor.close());
      feedbacktensors.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      weights.freeRef();
      RefUtil.freeRef(inputs);
      tfsession.freeRef();
      if (null != stateNames)
        stateNames.freeRef();
    }
  }
}
