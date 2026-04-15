package variable

// Square returns a variable representing x[0]^2.
func Square(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &SquareT{
			PowT{
				P: 2.0,
			},
		},
	}).First(x...)
}

// SquareT is the differentiable square operation.
type SquareT struct {
	PowT
}
