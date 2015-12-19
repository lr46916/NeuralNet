package neruralnet.layer.activation;

import java.io.Serializable;

import neruralnet.function.activation.ActivationFun;
import neruralnet.layer.StatelessLayer;

public class ActivationFunLayer extends StatelessLayer implements Serializable{

	private static final long serialVersionUID = 1424958538771748644L;
	
	private ActivationFun fun;
	private int n;
	
	public ActivationFunLayer(ActivationFun fun, int n) {
		super();
		this.fun = fun;
		this.n = n;
	}

	@Override
	public void setWeigths(double[] weights, int offset) {
		//nothing to do here
	}

	@Override
	public void apply(double[] inputs, double[] outputs) {
		assert(inputs.length == n && outputs.length == n);
		for(int i = 0; i < n; i++){
			outputs[i] = fun.apply(inputs[i]);
		}
	}

	@Override
	public double[] getWeights() {
		return null;
	}

	@Override
	public int getNumberOfInputs() {
		return n;
	}

	@Override
	public int getNumberOfOutputs() {
		return n;
	}

	@Override
	public int getNumberOfWeights() {
		return 0;
	}

}
