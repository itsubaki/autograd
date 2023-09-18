package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Add(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &AddT{}}).Apply(x...)[0]
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
		variable.Clone(gy[0]),
		variable.Clone(gy[0]),
	}
}
