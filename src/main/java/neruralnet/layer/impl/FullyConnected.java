package neruralnet.layer.impl;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;

import neruralnet.function.distribution.Distribution;
import neruralnet.layer.BPLayer;
import neruralnet.layer.Layer;
import neruralnet.layer.PrototypeLayer;

public class FullyConnected implements Serializable, PrototypeLayer, BPLayer {

	private static final long serialVersionUID = 3878314843105195424L;

	private double[][] weights;
	private int n, m;

	// backpropagation metadata
	private int[] iteratorData;

	public FullyConnected(int n, int m) {
		super();
		this.n = n;
		this.m = m;
		weights = new double[m][n + 1];
		iteratorData = new int[m];
		for (int i = 0; i < m; ++i) {
			iteratorData[i] = i;
		}
	}

	public FullyConnected(int n, int m, Distribution dist) {
		this(n, m);
		setWeigths(dist.getElements((n + 1) * m), 0);
	}

	@Override
	public void setWeigths(double[] weights, int offset) {
		assert ((n + 1) * m <= weights.length - offset);
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n + 1; j++) {
				this.weights[i][j] = weights[offset++];
			}
		}
	}

	@Override
	public void apply(double[] inputs, double[] outputs) {
		assert (outputs.length == m && inputs.length == n);
		for (int i = 0; i < m; i++) {
			outputs[i] = -weights[i][n];
			for (int j = 0; j < n; j++) {
				outputs[i] += weights[i][j] * inputs[j];
			}
		}
	}

	@Override
	public double[] getWeights() {
		double[] result = new double[(n + 1) * m];
		int count = 0;
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n + 1; ++j) {
				result[count++] = weights[i][j];
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
		weights[j][i] += value;
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
		return (n + 1) * m;
	}

	@Override
	public Iterable<Integer> connectedNeuronsIndexes(int inputIndex) {
		return () -> new FullyConnectedIterator();
	}

	@Override
	public PrototypeLayer duplicate() {
		FullyConnected res = new FullyConnected(n, m);
		return res;
	}

	@Override
	public boolean isStateful() {
		return false;
	}

	public static void main(String[] args) {

		Layer l = new FullyConnected(3, 1, (n) -> {
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

	private class FullyConnectedIterator implements Iterator<Integer> {
		private int index = 0;

		@Override
		public boolean hasNext() {
			return index != iteratorData.length;
		}

		@Override
		public Integer next() {
			return iteratorData[index++];
		}
	}

}
