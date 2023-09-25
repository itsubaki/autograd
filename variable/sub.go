package variable

import "github.com/itsubaki/autograd/vector"

// SubC returns a variable that c - x[0].
func SubC(c float64, x ...*Variable) *Variable {
	return (&Function{Forwarder: &SubT{}}).ApplyAndFirst(Const(c), x[0])
}

// Sub returns a variable that x[0] - x[1].
func Sub(x ...*Variable) *Variable {
	return (&Function{Forwarder: &SubT{}}).ApplyAndFirst(x...)
}

type SubT struct{}

func (f *SubT) Forward(x ...*Variable) []*Variable {
	y := vector.Sub(x[0].Data, x[1].Data)
	return []*Variable{
		New(y...),
	}
}

func (f *SubT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		gy[0],
		Neg(gy[0]), // -1.0 * gy
	}
}
