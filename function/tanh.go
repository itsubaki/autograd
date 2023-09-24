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

	y := vector.Tanh(x[0].Data)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *TanhT) Backward(gy ...*variable.Variable) []*variable.Variable {
	y := Tanh(f.x)
	return []*variable.Variable{
		Mul(gy[0], SubC(1.0, Mul(y, y))), // gy * (1-y^2)
	}
}
