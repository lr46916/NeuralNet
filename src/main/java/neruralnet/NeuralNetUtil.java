package neruralnet;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import neruralnet.function.activation.ActivationFun;
import neruralnet.function.distribution.Distribution;
import neruralnet.layer.Layer;
import neruralnet.layer.impl.ElmanLayer;
import neruralnet.layer.impl.FullyConnected;

public class NeuralNetUtil {
	public static StatelessLayeredNeuralNet createNetwork(String configuration, Distribution dist,
			ActivationFun... activationFuns) {
		String[] inputsOutputs = configuration.split("[x]");
		List<Layer> layers = new ArrayList<>();
		List<ActivationFun> activation = new ArrayList<>(activationFuns.length);
		for (int i = 0; i < inputsOutputs.length - 1; ++i) {
			int in = Integer.parseInt(inputsOutputs[i]);
			int out = Integer.parseInt(inputsOutputs[i + 1]);
			layers.add(new FullyConnected(in, out, dist));
			activation.add(activationFuns[i]);
		}

		return new StatelessLayeredNeuralNet(layers, activation);
	}

	public static StatefullLayeredNetwork createStatefullNetwork(String configuration,
			ActivationFun... activationFuns) {
		String[] structureAndContext = configuration.split("[_]");
		String[] inputsOutputs = structureAndContext[0].split("[x]");
		Set<Integer> contextLayers = new TreeSet<>();
		for (int i = 1; i < structureAndContext.length; ++i) {
			contextLayers.add(Integer.parseInt(structureAndContext[i]));
		}
		List<Layer> layers = new ArrayList<>();
		List<ActivationFun> activation = new ArrayList<>(activationFuns.length);
		for (int i = 0; i < inputsOutputs.length - 2; ++i) {
			int in = Integer.parseInt(inputsOutputs[i]);
			int out = Integer.parseInt(inputsOutputs[i + 1]);
			if (contextLayers.contains(i)) {
				layers.add(new ElmanLayer(in, out, activationFuns[i]));
			} else {
				layers.add(new FullyConnected(in, out));
			}
			activation.add(activationFuns[i]);
		}
		int in = Integer.parseInt(inputsOutputs[inputsOutputs.length - 2]);
		int out = Integer.parseInt(inputsOutputs[inputsOutputs.length - 1]);
		if (contextLayers.contains(inputsOutputs.length - 2)) {
			layers.add(new ElmanLayer(in, out, activationFuns[inputsOutputs.length - 2]));
		} else {
			layers.add(new FullyConnected(in, out));
		}
		activation.add(activationFuns[inputsOutputs.length - 2]);
		return new StatefullLayeredNetwork(layers, activation);
	}

	public static double[][] allocateMetadata(Layer[] layers, int n, int m) {
		double[][] networkMetadata = new double[layers.length][];

		int c = 0;
		int lastLayerOutNum = n;
		for (Layer l : layers) {
			networkMetadata[c++] = new double[l.getNumberOfOutputs()];
			if (lastLayerOutNum != l.getNumberOfInputs()) {
				throw new IllegalArgumentException("Inputs and outputs of layers do not match." + "at layer: " + (c - 1)
						+ " last layer outputs: " + lastLayerOutNum + " current inputs: " + l.getNumberOfInputs());
			} else {
				lastLayerOutNum = l.getNumberOfOutputs();
			}
		}
		if (lastLayerOutNum != m) {
			throw new IllegalArgumentException("Inputs and outputs of layers do not match." + "at layer: " + (c - 1)
					+ " last layer outputs: " + lastLayerOutNum + " network outputs: " + m);
		}
		return networkMetadata;
	}

}
