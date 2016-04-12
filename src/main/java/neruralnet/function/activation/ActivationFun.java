package neruralnet.function.activation;

import neruralnet.function.activation.impl.IdentityFunction;
import neruralnet.function.activation.impl.ReLU;
import neruralnet.function.activation.impl.SigmoidFunction;
import neruralnet.function.activation.impl.SigmoidFunctionApprox;
import neruralnet.function.activation.impl.TangHiperFunction;

public interface ActivationFun {
	public static final ActivationFunBP sigmoid = new SigmoidFunction();
	public static final ActivationFunBP sigmoidApprox = new SigmoidFunctionApprox();
	public static final ActivationFunBP tangh = new TangHiperFunction();
	public static final ActivationFunBP reLU = new ReLU();
	public static final ActivationFunBP identity = new IdentityFunction();

	public double apply(double value);
}
