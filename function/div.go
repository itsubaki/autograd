package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Div(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &DivT{}}).ApplyAndFirst(x...)
}

type DivT struct {
	x0, x1 *variable.Variable
}

func (f *DivT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x0, f.x1 = x[0], x[1]

	y := vector.Div(f.x0.Data, f.x1.Data)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *DivT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Div(gy[0], f.x1),
		Mul(Div(Neg(f.x0), Pow(2.0)(f.x1)), gy[0]), // -x0 / x1^2 * gy
	}
}
