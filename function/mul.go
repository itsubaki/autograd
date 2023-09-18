package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Mul(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &MulT{}}).Apply(x...)[0]
}

type MulT struct {
	x0, x1 variable.Data
}

func (f *MulT) Forward(x ...variable.Data) []variable.Data {
	f.x0, f.x1 = x[0], x[1]

	y := vector.Mul(f.x0, f.x1)
	return []variable.Data{y}
}

func (f *MulT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Mul(variable.New(f.x1...), gy[0]),
		Mul(variable.New(f.x0...), gy[0]),
	}
}
