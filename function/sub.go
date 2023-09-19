package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

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
		Mul(variable.Const(-1.0), gy[0]), // -1.0 * gy
	}
}
