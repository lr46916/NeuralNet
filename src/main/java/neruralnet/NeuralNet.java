package neruralnet;

import neruralnet.layer.Layer;

public class NeuralNet implements Layer {

	private int n;
	private int m;
	private int numOfWeights;
	private Layer[] layers;
	private double[][] innerLayerOut;

	public NeuralNet(Layer... layers) {
		this.layers = layers;

		n = this.layers[0].getNumberOfInputs();
		m = this.layers[this.layers.length - 1].getNumberOfOutputs();

		this.innerLayerOut = new double[this.layers.length][];

		int c = 0;
		numOfWeights = 0;
		int lastLayerOutNum = n;
		for (Layer l : this.layers) {
			numOfWeights += l.getNumberOfWeights();
			this.innerLayerOut[c++] = new double[l.getNumberOfOutputs()];
			if (lastLayerOutNum != l.getNumberOfInputs()) {
				throw new IllegalArgumentException();
			} else {
				lastLayerOutNum = l.getNumberOfOutputs();
			}
		}
	}

	public void setWeigths(double[] weights, int offset) {
		assert (weights.length == numOfWeights);
		for (Layer l : layers) {
			l.setWeigths(weights, offset);
			offset += l.getNumberOfWeights();
		}
	}

	public void apply(double[] inputs, double[] outputs) {
		assert (inputs.length == n && outputs.length == m);

		layers[0].apply(inputs, innerLayerOut[0]);

		for (int i = 1; i < layers.length; i++) {
			layers[i].apply(innerLayerOut[i - 1], innerLayerOut[i]);
		}

		for (int i = 0; i < m; i++) {
			outputs[i] = innerLayerOut[innerLayerOut.length - 1][i];
		}
	}

	// quite costly huh?
	public double[] getWeights() {
		double[] res = new double[numOfWeights];
		int c = 0;
		for(Layer l : layers) {
			for(double w : l.getWeights()) {
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

}
