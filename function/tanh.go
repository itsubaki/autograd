package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Tanh(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &TanhT{}}).ApplyAndFirst(x...)
}

type TanhT struct {
	x *variable.Variable
}

func (f *TanhT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]

	y := vector.Tanh(f.x.Data)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *TanhT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Mul(SubC(1.0, Pow(2.0)(Tanh(f.x))), gy[0]), // (1-y^2) * gy
	}
}
