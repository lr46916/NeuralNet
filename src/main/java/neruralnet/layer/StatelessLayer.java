package neruralnet.layer;

public abstract class StatelessLayer implements Layer {

	@Override
	public boolean isStateful() {
		return false;
	}

}
