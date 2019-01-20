package com.simiacryptus.mindseye.tensorflow;

import com.simiacryptus.util.Util;
import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.util.List;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;

public class LabelingNetwork {
  private final byte[] protobufSrc;
  private final List<String> labels;

  public byte[] getProtobufSrc() {
    return protobufSrc;
  }

  public List<String> getLabels() {
    return labels;
  }

  public LabelingNetwork()  {
    try(ZipFile zipFile = new ZipFile(Util.cacheFile(new URI("https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip")))) {
      protobufSrc = IOUtils.toByteArray(zipFile.getInputStream(zipFile.getEntry("tensorflow_inception_graph.pb")));
      labels = IOUtils.readLines(zipFile.getInputStream(zipFile.getEntry("imagenet_comp_graph_label_strings.txt")), "UTF-8");
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
  }
}
