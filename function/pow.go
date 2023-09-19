package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Pow(c float64) func(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &PowT{C: c}}).ApplyS
}

type PowT struct {
	C float64
	x *variable.Variable
}

func (f *PowT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]

	y := vector.Pow(f.x.Data, f.C)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *PowT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Mul(variable.Const(f.C), Mul(Pow(f.C-1)(f.x), gy[0])), // c * x^(c-1) * gy
	}
}
