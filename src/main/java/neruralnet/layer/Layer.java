package neruralnet.layer;

public interface Layer {
	public void setWeigths(double[] weights, int offset);
	public void apply(double[] inputs, double[] outputs);
	public double[] getWeights();
	public int getNumberOfInputs();
	public int getNumberOfOutputs();
	public int getNumberOfWeights();
}
