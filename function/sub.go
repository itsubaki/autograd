package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

// SubC returns a variable that c - x[0].
func SubC(c float64, x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &SubT{}}).ApplyAndFirst(variable.Const(c), x[0])
}

// Sub returns a variable that x[0] - x[1].
func Sub(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &SubT{}}).ApplyAndFirst(x...)
}

type SubT struct{}

func (f *SubT) Forward(x ...*variable.Variable) []*variable.Variable {
	y := vector.Sub(x[0].Data, x[1].Data)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *SubT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		gy[0],
		Neg(gy[0]), // -1.0 * gy
	}
}
