package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Tanh(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &TanhT{}}).ApplyAndFirst(x...)
}

type TanhT struct {
	y *variable.Variable
}

func (f *TanhT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.y = variable.New(vector.Tanh(x[0].Data)...)
	return []*variable.Variable{f.y}
}

func (f *TanhT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Mul(AddC(1.0, Neg(Pow(2.0)(f.y))), gy[0]), // (1-y^2) * gy
	}
}
