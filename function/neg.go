package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Neg(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &NegT{}}).ApplyAndFirst(x...)
}

type NegT struct{}

func (f *NegT) Forward(x ...*variable.Variable) []*variable.Variable {
	y := vector.MulC(x[0].Data, -1.0)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *NegT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Neg(gy[0]),
	}
}
