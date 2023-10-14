package variable

func Square(x ...*Variable) *Variable {
	return (&Function{Forwarder: &SquareT{PowT{C: 2.0}}}).First(x...)
}

type SquareT struct {
	PowT
}
