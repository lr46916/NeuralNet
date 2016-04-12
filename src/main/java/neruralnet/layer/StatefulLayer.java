package neruralnet.layer;

public abstract class StatefulLayer implements Layer {
	public abstract double[] getContext();

	public abstract void setContext(double[] context, int offset);

	public abstract int getContextSize();

	@Override
	public boolean isStateful() {
		return true;
	}
}