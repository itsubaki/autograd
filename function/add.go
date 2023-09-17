package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Add(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &AddT{}}).Apply(x...)[0]
}

type AddT struct{}

func (f *AddT) Forward(x ...variable.Data) []variable.Data {
	y := vector.Add(x[0], x[1])
	return []variable.Data{y}
}

func (f *AddT) Backward(gy ...variable.Data) []variable.Data {
	return []variable.Data{gy[0], gy[0]}
}
