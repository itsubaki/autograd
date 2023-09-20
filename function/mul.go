package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func MulC(c float64, x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &MulT{}}).ApplyAndFirst(variable.Const(c), x[0])
}

func Mul(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &MulT{}}).ApplyAndFirst(x...)
}

type MulT struct {
	x0, x1 *variable.Variable
}

func (f *MulT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x0, f.x1 = x[0], x[1]

	y := vector.Mul(f.x0.Data, f.x1.Data)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *MulT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Mul(f.x1, gy[0]), // x1 * gy
		Mul(f.x0, gy[0]), // x0 * gy
	}
}
