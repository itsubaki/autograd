package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Neg(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &NegT{}}).ApplyS(x...)
}

type NegT struct {
	x *variable.Variable
}

func (f *NegT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]

	y := vector.MulC(f.x.Data, -1.0)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *NegT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Neg(gy[0]),
	}
}
