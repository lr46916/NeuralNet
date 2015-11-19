package neruralnet;

public interface NeuralNet {
	public void setWeigths(double[] weights);
	public void apply(double[] inputs, double[] outputs);
	public double[] getWeights();
	public int getNumberOfInputs();
	public int getNumberOfOutputs();
	public int getNumberOfWeights();
}
