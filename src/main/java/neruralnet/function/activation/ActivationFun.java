package neruralnet.function.activation;

import neruralnet.function.activation.impl.SigmoidFunction;
import neruralnet.function.activation.impl.TangHiperFunction;

public interface ActivationFun {
	public static ActivationFun sigmoid = new SigmoidFunction();
	public static ActivationFun tangh = new TangHiperFunction();

	public double apply(double value);
}
