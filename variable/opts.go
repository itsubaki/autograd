package variable

// Opts configures backward propagation behavior.
type Opts struct {
	RetainGrad  bool
	CreateGraph bool
}

// NoRetainGrad reports whether output gradients should be discarded after use.
func NoRetainGrad(opts ...Opts) bool {
	return !HasRetainGrad(opts...)
}

// HasRetainGrad reports whether output gradients should be retained.
func HasRetainGrad(opts ...Opts) bool {
	if len(opts) > 0 && opts[0].RetainGrad {
		return true
	}

	return false
}

// NoCreateGraph reports whether backpropagation should avoid creating a new graph.
func NoCreateGraph(opts ...Opts) bool {
	return !HasCreateGraph(opts...)
}

// HasCreateGraph reports whether backpropagation should create a new graph.
func HasCreateGraph(opts ...Opts) bool {
	if len(opts) > 0 && opts[0].CreateGraph {
		return true
	}

	return false
}
