package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Pow(c float64) func(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &PowT{C: c}}).ApplyAndFirst
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
		Mul(gy[0], MulC(f.C, Pow(f.C-1)(f.x))), // gy * c * x^(c-1)
	}
}
