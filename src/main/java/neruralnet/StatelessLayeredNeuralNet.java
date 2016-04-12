package neruralnet;

import java.io.Serializable;
import java.util.List;

import neruralnet.function.activation.ActivationFun;
import neruralnet.layer.Layer;

public class StatelessLayeredNeuralNet implements Serializable, Layer {

	private static final long serialVersionUID = 7405336452347186813L;

	private int n;
	private int m;
	private int numOfWeights;
	private Layer[] layers;
	private ActivationFun[] activation;
	private double[][] innerLayerOut;

	public StatelessLayeredNeuralNet(Layer[] layers, ActivationFun[] activation) {
		this.activation = activation;
		initalize(layers);
	}

	public StatelessLayeredNeuralNet(List<Layer> layers, List<ActivationFun> activation) {
		Layer[] layerArray = new Layer[layers.size()];
		for (int i = 0; i < layers.size(); ++i) {
			layerArray[i] = layers.get(i);
		}
		this.activation = new ActivationFun[activation.size()];
		for (int i = 0; i < activation.size(); ++i) {
			this.activation[i] = activation.get(i);
		}
		initalize(layerArray);
	}

	private void initalize(Layer[] layers) {
		this.layers = layers;
		n = this.layers[0].getNumberOfInputs();
		m = this.layers[this.layers.length - 1].getNumberOfOutputs();

		if (layers.length != activation.length) {
			throw new IllegalArgumentException(
					"Number of layers and activation functions is not the same. Each layer must have an activoation function.");
		}

		this.innerLayerOut = NeuralNetUtil.allocateMetadata(layers, n, m);

		numOfWeights = 0;

		for (Layer l : layers)
			numOfWeights += l.getNumberOfWeights();
	}

	public Layer[] getLayers() {
		return layers;
	}

	public ActivationFun[] getActivations() {
		return activation;
	}

	public void setWeigths(double[] weights, int offset) {
		assert (weights.length == numOfWeights);
		for (Layer l : layers) {
			l.setWeigths(weights, offset);
			offset += l.getNumberOfWeights();
		}
	}

	private static void applyActivation(double[] target, ActivationFun activation) {
		for (int i = 0; i < target.length; ++i) {
			target[i] = activation.apply(target[i]);
		}
	}

	public void apply(double[] inputs, double[] outputs) {
		assert (inputs.length == n && outputs.length == m);
		layers[0].apply(inputs, innerLayerOut[0]);
		applyActivation(innerLayerOut[0], activation[0]);
		for (int i = 1; i < layers.length; i++) {
			layers[i].apply(innerLayerOut[i - 1], innerLayerOut[i]);
			applyActivation(innerLayerOut[i], activation[i]);
		}
		for (int i = 0; i < m; i++) {
			outputs[i] = innerLayerOut[innerLayerOut.length - 1][i];
		}
	}

	// quite costly huh?
	public double[] getWeights() {
		double[] res = new double[numOfWeights];
		int c = 0;
		for (Layer l : layers) {
			for (double w : l.getWeights()) {
				res[c++] = w;
			}
		}
		return res;
	}

	public int getNumberOfInputs() {
		return n;
	}

	public int getNumberOfOutputs() {
		return m;
	}

	public int getNumberOfWeights() {
		return numOfWeights;
	}

	@Override
	public boolean isStateful() {
		return false;
	}

}
