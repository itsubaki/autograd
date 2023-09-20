package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func AddC(c float64, x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &AddT{}}).ApplyAndFirst(variable.Const(c), x[0])
}

func Add(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &AddT{}}).ApplyAndFirst(x...)
}

type AddT struct{}

func (f *AddT) Forward(x ...*variable.Variable) []*variable.Variable {
	y := vector.Add(x[0].Data, x[1].Data)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *AddT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		gy[0],
		gy[0],
	}
}
