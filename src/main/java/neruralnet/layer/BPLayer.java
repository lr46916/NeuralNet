package neruralnet.layer;

public interface BPLayer extends Layer {
	public double getWeight(int i, int j);
	public void updateWeight(int i, int j, double value);

	/**
	 * Method to provide iterator over all indexes of output neurons that are
	 * connected to given input index.
	 * 
	 * @param inputIndex index of input for which connection data is required.
	 * @return iterator over all output indexes to which input at given index is connected.
	 */
	public Iterable<Integer> connectedNeuronsIndexes(int inputIndex);
}
