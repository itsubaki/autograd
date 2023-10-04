package variable

type Opts struct {
	RetainGrad  bool
	CreateGraph bool
}

func HasRetainGrad(opts ...Opts) bool {
	if len(opts) > 0 && opts[0].RetainGrad {
		return true
	}

	return false
}

func HasCreateGraph(opts ...Opts) bool {
	if len(opts) > 0 && opts[0].CreateGraph {
		return true
	}

	return false
}
