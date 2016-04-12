package neruralnet.function.activation;

public interface ActivationFunBP extends ActivationFun {
	public double gradient(double x);
}
