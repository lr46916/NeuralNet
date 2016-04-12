package neruralnet.layer.impl;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;

import neruralnet.function.activation.ActivationFun;
import neruralnet.function.distribution.Distribution;
import neruralnet.layer.BPLayer;
import neruralnet.layer.Layer;
import neruralnet.layer.PrototypeLayer;
import neruralnet.layer.StatefulLayer;

public class ElmanLayer extends StatefulLayer implements Serializable, PrototypeLayer, BPLayer {

	private static final long serialVersionUID = 3878314843105195424L;

	private double[][] weights;
	private int n, m;
	private ActivationFun activation;
	private double[] context;

	public ElmanLayer(int n, int m, ActivationFun activation) {
		super();
		this.n = n;
		this.m = m;
		this.activation = activation;
		this.context = new double[m];
		weights = new double[m][n + m + 1];
	}

	public ElmanLayer(int n, int m, ActivationFun activation, Distribution dist) {
		this(n, m, activation);
		setWeigths(dist.getElements((n + m + 1) * m), 0);
	}

	@Override
	public void setWeigths(double[] weights, int offset) {
		assert ((n + 1) * m <= weights.length - offset);
		for (int j = 0; j < m; ++j) {
			for (int i = 0; i < n + 1; i++) {
				this.weights[j][i] = weights[offset++];
			}
		}
	}

	@Override
	public void apply(double[] inputs, double[] outputs) {
		assert (outputs.length == m && inputs.length == n);
		for (int i = 0; i < m; i++) {
			outputs[i] = -weights[i][n + m];
			for (int j = 0; j < n; j++) {
				outputs[i] += weights[i][j] * inputs[j];
			}
			for (int j = n; j < n + m; ++j) {
				outputs[i] += weights[i][j] * context[j - n];
			}
		}
		for (int i = 0; i < m; ++i) {
			context[i] = activation.apply(outputs[i]);
		}
	}

	@Override
	public double[] getWeights() {
		double[] result = new double[(n + 1) * m];
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n + 1; ++j) {
				result[i * m + j] = weights[i][j];
			}
		}
		return result;
	}

	@Override
	public double getWeight(int i, int j) {
		return weights[j][i];
	}
	
	@Override
	public void updateWeight(int i, int j, double value) {
		weights[i][j] += value;
	}

	@Override
	public int getNumberOfInputs() {
		return n;
	}

	@Override
	public int getNumberOfOutputs() {
		return m;
	}

	@Override
	public int getNumberOfWeights() {
		return (n + m + 1) * m;
	}

	@Override
	public Iterable<Integer> connectedNeuronsIndexes(int inputIndex) {
		throw new IllegalAccessError("Not yet implemented");
	}

	@Override
	public PrototypeLayer duplicate() {
		ElmanLayer res = new ElmanLayer(n, m, activation);
		return res;
	}

	@Override
	public double[] getContext() {
		return context;
	}

	@Override
	public void setContext(double[] context, int offset) {
		for (int i = 0; i < m; ++i) {
			this.context[i] = context[offset++];
		}
	}

	@Override
	public int getContextSize() {
		return m;
	}

	public static void main(String[] args) {

		Layer l = new ElmanLayer(3, 1, (x) -> x > 0 ? 1 : 0, (n) -> {
			double[] res = new double[n];
			res[0] = 4;
			for (int i = 1; i < n; i++)
				res[i] = 1;
			return res;
		});

		double[] inputs = new double[] { 1, 1, 2.01 };
		double[] outputs = new double[1];

		l.apply(inputs, outputs);

		System.out.println(Arrays.toString(outputs));

	}

}
